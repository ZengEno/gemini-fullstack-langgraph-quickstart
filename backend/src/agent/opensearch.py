import os
import json
from pydantic import BaseModel
from typing import List, Dict, Any, Literal, Optional
import requests

class SearchResultItem(BaseModel):
    link: str
    title: str
    content: str
    snippet: str
    position: Optional[str] = None

class Usage(BaseModel):
    rewrite_model_output_tokens: int = 0
    rewrite_model_total_tokens: int = 0
    rewrite_model_input_tokens: int = 0
    search_count: int = 0

class SearchResult(BaseModel):
    search_result: List[SearchResultItem]

class OpenSearchResponse(BaseModel):
    request_id: str
    latency: float
    usage: Usage
    result: SearchResult

def parse_opensearch_response(response: Dict[str, Any]) -> OpenSearchResponse:
    """
    Parse the OpenSearch service response.
    
    Args:
        response (Dict[str, Any]): The raw response from the OpenSearch service
        
    Returns:
        OpenSearchResponse: Parsed response object
        
    Raises:
        ValueError: If the response contains an error or cannot be parsed
    """
    # Check if it's an error response
    if "code" in response and "http_code" in response and "message" in response:
        raise ValueError(f"OpenSearch API Error: {response['code']} - {response['message']}")
    
    # Parse as success response
    try:
        return OpenSearchResponse(**response)
    except Exception as e:
        raise ValueError(f"Failed to parse OpenSearch response: {str(e)}")

def call_opensearch_service(
    query: str,
    history: List[Dict[str, str]],
    way: Literal["fast", "normal", "full"] = "fast",
    query_rewrite: bool = True,
    top_k: int = 5,
    content_type: Literal["snippet", "summary"] = "snippet"
) -> OpenSearchResponse:
    """
    Call the Aliyun OpenSearch service with the given parameters. 
    Check this link: https://help.aliyun.com/zh/open-search/search-platform/developer-reference/
    
    Args:
        query (str): The search query
        history (List[Dict[str, str]]): Conversation history in the format [{"role": str, "content": str}, ...]
        way (Literal["fast", "normal", "full"]): Search method, defaults to "fast"
        query_rewrite (bool): Whether to rewrite the query using LLM, defaults to True
        top_k (int): Number of results to return, defaults to 5
        content_type (Literal["snippet", "content"]): Type of content to return, defaults to "snippet"
        
    Returns:
        OpenSearchResponse: The parsed response from the OpenSearch service
        
    Raises:
        ValueError: If required environment variables are not set or if the API returns an error
        requests.RequestException: If the API call fails
    """
    # Get environment variables
    api_key = os.getenv("ALIYUN_OPENSEARCH_API_KEY")
    host = os.getenv("ALIYUN_OPENSEARCH_HOST")
    workspace_name = os.getenv("ALIYUN_OPENSEARCH_WORKSPACE_NAME")
    
    # Validate environment variables
    if not all([api_key, host, workspace_name]):
        raise ValueError("Missing required environment variables: ALIYUN_OPENSEARCH_API_KEY, ALIYUN_OPENSEARCH_HOST, or ALIYUN_OPENSEARCH_WORKSPACE_NAME")
    
    # Construct the URL
    url = f"http://{host}/v3/openapi/workspaces/{workspace_name}/web-search/ops-web-search-001"
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Prepare request body
    payload = {
        "history": history,
        "query": query,
        "way": way,
        "query_rewrite": query_rewrite,
        "top_k": top_k,
        "content_type": content_type
    }
    
    # Make the request
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    # Parse and return the response
    return parse_opensearch_response(response.json())