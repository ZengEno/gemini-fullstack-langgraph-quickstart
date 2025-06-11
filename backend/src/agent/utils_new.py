import re
from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(search_result: List[Any], id: int) -> Dict[str, str]:
    """
    Create a map of the search result urls to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"url_id/"
    urls = [search_result_item.link for search_result_item in search_result]

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def get_modified_text(search_result: List[Any], resolved_map: Dict[str, str]) -> str:
    """
    Get the modified text from the search result.
    """
    modified_text = ""
    for search_result_item in search_result:
        modified_text += f"{search_result_item.snippet} [{search_result_item.title}]({resolved_map[search_result_item.link]})\n"
    return modified_text


def format_citation_sources(search_result: List[Any], resolved_map: Dict[str, str]): 
    citation_sources = []
    for search_result_item in search_result:
        label = search_result_item.title
        short_url = resolved_map[search_result_item.link]
        value = search_result_item.link
        citation_sources.append({"label": label,
                                 "short_url": short_url, 
                                 "value": value})
    return citation_sources



def extract_json_content(text: str) -> str:
    """Extract JSON content from a string that starts with ```json and ends with ```.
    
    Args:
        text: String containing JSON content wrapped in ```json and ```
        
    Returns:
        Extracted JSON content as a string
    """
    # First check if the string starts with ```json and ends with ```
    if not (text.startswith("```json") and text.endswith("```")):
        return ""
        
    # Remove the ```json and ``` markers
    text = text[7:-3].strip()
    
    # Find content between first { and last }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return ""