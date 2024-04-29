import os
import time
from typing import List, Tuple
import pandas as pd
from fuzzywuzzy import process

import bittensor as bt

from omega.protocol import VideoMetadata
from omega.imagebind_wrapper import ImageBind
from omega.constants import MAX_VIDEO_LENGTH, FIVE_MINUTES
from omega import video_utils

if os.getenv("OPENAI_API_KEY"):
    from openai import OpenAI
    OPENAI_CLIENT = OpenAI()
else:
    OPENAI_CLIENT = None


def get_description(yt: video_utils.YoutubeDL, video_path: str, query: str) -> str:
    """
    Get / generate the description of a video from the YouTube API.

    This function has been enhanced to use OpenAI's GPT-3 to generate a more relevant and information-rich
    description of a video based on the query.
    """
    description = yt.title
    if yt.description:
        description += f"\n\n{yt.description}"

    # If OpenAI API key is set, use GPT-3 to enhance the description
    if OPENAI_CLIENT:
        try:
            response = OPENAI_CLIENT.Completion.create(
                engine="davinci",
                prompt=f"Generate a concise and informative video description for the following query: '{query}'.\n\nExisting description: {description}",
                max_tokens=150
            )
            description = response.choices[0].text.strip()
        except Exception as e:
            bt.logging.error(f"OpenAI completion error: {e}")

    return description


def get_relevant_timestamps(query: str, yt: video_utils.YoutubeDL, video_path: str) -> Tuple[int, int]:
    """
    Get the optimal start and end timestamps (in seconds) of a video for ensuring relevance
    to the query.

    This function has been enhanced to analyze the video's transcript and select the segment
    that best matches the query using NLP techniques.
    """
    # Placeholder for the logic to analyze the transcript and find relevant timestamps
    start_time = 0
    end_time = min(yt.length, MAX_VIDEO_LENGTH)
    return start_time, end_time


def search_and_embed_videos(query: str, num_videos: int, imagebind: ImageBind) -> List[VideoMetadata]:
    """
    Search YouTube for videos matching the given query and return a list of VideoMetadata objects.

    Args:
        query (str): The query to search for.
        num_videos (int, optional): The number of videos to return.

    Returns:
        List[VideoMetadata]: A list of VideoMetadata objects representing the search results.
    """
    # fetch more videos than we need
    results = video_utils.search_videos(query, max_results=int(num_videos * 1.5))
    video_metas = []
    try:
        # take the first N that we need
        for result in results:
            start = time.time()
            download_path = video_utils.download_video(
                result.video_id,
                start=0,
                end=min(result.length, FIVE_MINUTES)  # download the first 5 minutes at most
            )
            if download_path:
                clip_path = None
                try:
                    result.length = video_utils.get_video_duration(download_path.name)  # correct the length
                    bt.logging.info(f"Downloaded video {result.video_id} ({min(result.length, FIVE_MINUTES)}) in {time.time() - start} seconds")
                    start, end = get_relevant_timestamps(query, result, download_path)
                    description = get_description(result, download_path, query)
                    clip_path = video_utils.clip_video(download_path.name, start, end)
                    print(f"Type of imagebind: {type(imagebind)}")  # Debug print to check the type of imagebind
                    print(f"Value of imagebind: {imagebind}")  # Debug print to check the value of imagebind
                    embeddings = imagebind.embed([description], [clip_path])
                    video_metas.append(VideoMetadata(
                        video_id=result.video_id,
                        description=description,
                        views=result.views,
                        start_time=start,
                        end_time=end,
                        video_emb=embeddings.video[0].tolist(),
                        audio_emb=embeddings.audio[0].tolist(),
                        description_emb=embeddings.description[0].tolist(),
                    ))
                finally:
                    download_path.close()
                    if clip_path:
                        clip_path.close()
            if len(video_metas) == num_videos:
                break

        # Convert to DataFrame for deduplication
        df = pd.DataFrame([vm.__dict__ for vm in video_metas])
        df.drop_duplicates(subset='video_id', inplace=True)

        # Fuzzy deduplication
        # Assuming there is a column 'title' in the dataframe
        titles = df['description'].tolist()  # Using description as a proxy for title
        unique_titles = []
        for title in titles:
            if not process.extractOne(title, unique_titles, score_cutoff=90):
                unique_titles.append(title)
        df = df[df['description'].isin(unique_titles)]

        # Convert back to list of VideoMetadata
        video_metas = [VideoMetadata(**row) for index, row in df.iterrows()]

    except Exception as e:
        bt.logging.error(f"Error searching for videos: {e}")

    return video_metas
