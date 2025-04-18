from mcp.server.fastmcp import FastMCP
from parse_content import extract_url_content


mcp = FastMCP("urlParseContent")


@mcp.tool()
def url_parse_content(url: str, type: str) -> str:
    """
    You are a content processor that retrieves and processes files from provided URLs.

    Args:
        url (string): The  URL to the file. Supported formats include PDF, JPG, and PNG.
        type (string): The desired output format. Options are: text – Extract and return text content. json – Return structured data extracted from the content. ￼
    """
    try:
        # Convert the URL to a string
        if type== None:
            type = "text"
        # Use the extraction function
        result = extract_url_content(url, type)

        # For non-API formats, wrap the result in API format
        if type != "api":
            return f"""
                code: 200,
                msg: Success,
                data: {result}
            """

        # For API format, result is already in the correct format
        return result

    except Exception as e:
        # Return error in API format
        return f"""
            code: 500,
            msg: Error: {str(e)},
            data: None
        """


if __name__ == '__main__':
    mcp.run(transport="stdio")