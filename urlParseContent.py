from mcp.server.fastmcp import FastMCP
import os
import tempfile
import httpx
from urllib.parse import urlparse
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import io
import pypdf
import re
import argparse
import json

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


def download_file(url):
    """Download file from URL and return the content as bytes"""
    response = httpx.get(url, follow_redirects=True)
    response.raise_for_status()  # Raise an exception for 4XX/5XX responses
    return response.content


def get_file_extension(url):
    """Extract file extension from URL"""
    path = urlparse(url).path
    return os.path.splitext(path)[1].lower()


def extract_text_from_pdf(content):
    """Extract text from PDF content"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        text = ""
        with open(temp_file_path, 'rb') as file:
            pdf = pypdf.PdfReader(file)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip()
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def preprocess_image(image):
    """Apply preprocessing to improve OCR accuracy"""
    # Convert to grayscale
    image = image.convert('L')

    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Apply sharpening filter
    image = image.filter(ImageFilter.SHARPEN)

    # Apply noise reduction
    image = image.filter(ImageFilter.MedianFilter(size=3))

    # Resize for better OCR if image is too small
    if image.width < 1000 or image.height < 1000:
        ratio = max(1000 / image.width, 1000 / image.height)
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)

    return image


def extract_text_from_image(content, file_type):
    """Extract text from image content using enhanced OCR"""
    image = Image.open(io.BytesIO(content))

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Use multiple OCR configurations and combine results
    # Try different PSM modes for better detection
    text_results = []

    # PSM 3 - Fully automatic page segmentation, but no OSD (default)
    text_results.append(pytesseract.image_to_string(processed_image, config='--psm 3'))

    # PSM 6 - Assume a single uniform block of text
    text_results.append(pytesseract.image_to_string(processed_image, config='--psm 6'))

    # PSM 4 - Assume a single column of text of variable sizes
    text_results.append(pytesseract.image_to_string(processed_image, config='--psm 4'))

    # Combine and deduplicate results
    combined_text = "\n".join(text_results)

    return combined_text


def format_content_for_llm(text):
    """Format extracted text for better understanding by large language models"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Split into paragraphs by multiple newlines
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Remove duplicate paragraphs
    unique_paragraphs = []
    for p in paragraphs:
        if p not in unique_paragraphs:
            unique_paragraphs.append(p)

    # Detect headings (often all caps or followed by colon)
    structured_content = []
    for p in unique_paragraphs:
        if p.isupper() or re.match(r'^[A-Z][^a-z]*:', p):
            # This is likely a heading
            structured_content.append({"type": "heading", "content": p})
        elif re.match(r'^\d+\.\s', p):
            # This is likely a numbered list item
            structured_content.append({"type": "list_item", "content": p})
        elif len(p.split()) < 5:
            # Short text is likely a label or heading
            structured_content.append({"type": "label", "content": p})
        else:
            # Regular paragraph
            structured_content.append({"type": "paragraph", "content": p})

    # Format into a clean text with proper structure
    formatted_text = ""
    for item in structured_content:
        if item["type"] == "heading":
            formatted_text += f"\n## {item['content']}\n\n"
        elif item["type"] == "label":
            formatted_text += f"\n### {item['content']}\n"
        elif item["type"] == "list_item":
            formatted_text += f"- {item['content']}\n"
        else:
            formatted_text += f"{item['content']}\n\n"

    return formatted_text.strip()


def format_api_response(content, success=True, error_message=None):
    """
    Format the extracted content as a standardized API response
    with code, msg, and data fields

    Args:
        content: The extracted content
        success: Boolean indicating if the operation was successful
        error_message: Error message if operation failed

    Returns:
        Dictionary with standardized API response format
    """
    if success:
        response = {
            "code": 200,
            "msg": "Success",
            "data": content
        }
    else:
        response = {
            "code": 500,
            "msg": error_message or "An error occurred",
            "data": None
        }

    return response


def extract_content_from_url(url):
    """Main function to extract content from a URL"""
    # Get file extension
    extension = get_file_extension(url)

    if not extension:
        return format_api_response(None, False, "Unable to determine file type from URL")

    try:
        # Download the file
        content = download_file(url)

        # Process based on file type
        if extension == '.pdf':
            raw_text = extract_text_from_pdf(content)
        elif extension in ['.png', '.jpg', '.jpeg']:
            raw_text = extract_text_from_image(content, extension[1:])
        else:
            return format_api_response(None, False, f"Unsupported file type: {extension}")

        # Format content for better understanding
        formatted_text = format_content_for_llm(raw_text)

        # Create structured content object
        structured_content = {
            "raw_text": raw_text,
            "formatted_text": formatted_text,
            "sections": formatted_text.split("\n\n"),
            "file_type": extension[1:],
            "url": url
        }

        return format_api_response(structured_content)
    except Exception as e:
        return format_api_response(None, False, f"Error processing URL: {str(e)}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract text from PDF or image URLs')
    parser.add_argument('url', nargs='?', help='URL of the PDF or image file')
    parser.add_argument('-o', '--output', help='Output file to save extracted text')
    parser.add_argument('--format', choices=['text', 'json', 'api'], default='text',
                        help='Output format (text, json, or api)')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # If no URL provided via command line, prompt for it
    if args.url:
        url = args.url
    else:
        url = input("Enter URL of PDF or image (PNG/JPG) file: ")

    result = extract_content_from_url(url)

    # Format output based on user preference
    if args.format == 'api':
        # Already in API format
        output = json.dumps(result, ensure_ascii=False, indent=2)
    elif args.format == 'json':
        # For backward compatibility
        if isinstance(result, dict) and 'data' in result:
            # Extract data from API response
            data = result['data']
            content = data.get('formatted_text', '')
        else:
            # Legacy format handling (shouldn't happen in updated code)
            content = str(result)

        output = json.dumps({
            "url": url,
            "content": content,
            "sections": content.split("\n\n") if content else []
        }, ensure_ascii=False, indent=2)
    else:
        # Text format - extract the formatted text only
        if isinstance(result, dict) and 'data' in result:
            data = result['data']
            if data and 'formatted_text' in data:
                output = data['formatted_text']
            else:
                output = f"Error: {result.get('msg', 'Unknown error')}"
        else:
            # Legacy format handling
            output = str(result)

    # Output to file if specified, otherwise to console
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Content extracted and saved to {args.output}")
    else:
        print("\nExtracted Content:")
        print("-" * 50)
        print(output)


# Add a function to use as a module API
def extract_url_content(url, output_format='text'):
    """
    Extract content from a URL and return it in the specified format

    Args:
        url: URL of PDF or image file
        output_format: 'api', 'json', or 'text'

    Returns:
        Formatted content based on output_format parameter
    """
    result = extract_content_from_url(url)

    if output_format == 'api':
        return result
    elif output_format == 'json':
        if isinstance(result, dict) and 'data' in result:
            data = result['data']
            if data:
                content = data.get('formatted_text', '')
            else:
                content = ''
        else:
            content = str(result)

        return {
            "url": url,
            "content": content,
            "sections": content.split("\n\n") if content else []
        }
    else:  # text
        if isinstance(result, dict) and 'data' in result:
            data = result['data']
            if data and 'formatted_text' in data:
                return data['formatted_text']
            else:
                return f"Error: {result.get('msg', 'Unknown error')}"
        else:
            return str(result)


if __name__ == '__main__':
    mcp.run(transport="stdio")