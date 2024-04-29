import sys
sys.path.append('/home/ubuntu/omegalabs-bittensor-subnet-main/omega')
from miner_utils import search_and_embed_videos
from imagebind_wrapper import ImageBind

# Simulate validator queries
queries = [
    "How to train neural networks",
    "Latest advancements in artificial intelligence",
    "Understanding blockchain technology",
    "Exploring the depths of the ocean",
    "Space exploration and the future of space travel"
]

# Function to test the scraping method
def test_scrape(queries, num_videos=5):
    imagebind = ImageBind()
    for query in queries:
        print(f"Query: {query}")
        results = search_and_embed_videos(query, num_videos, imagebind)
        print(f"Results for '{query}':")
        for result in results:
            print(f"Video ID: {result.video_id}, Description: {result.description}, Views: {result.views}, Start Time: {result.start_time}, End Time: {result.end_time}\n")

# Run the test
if __name__ == "__main__":
    test_scrape(queries)
