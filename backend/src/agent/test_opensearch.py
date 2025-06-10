import os
from opensearch import call_opensearch_service
from dotenv import load_dotenv

load_dotenv()

def test_request():
    # Valid conversation history
    history = [
        {"role": "system", "content": "你是一个机器人助手"},
    ]
    
    # Valid query
    query = "什么是宇宙大爆炸"
    
    try:
        response = call_opensearch_service(
            query=query,
            history=history,
            way="fast",
            query_rewrite=True,
            top_k=5,
            content_type="snippet"
        )
        print("Success Response:")
        print(response.result.search_result)
        print(type(response.result.search_result))
    except ValueError as e:
        print(f"OpenSearch error: {e}")
    except Exception as e:
        print(f"Other error: {e}")


if __name__ == "__main__":
    print("Testing valid request...")
    test_request()

