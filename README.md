## 服务介绍

这是一个用于解析pdf内容的mcp server，用于解析 pdf、png、jpg等格式，并整理成大模型所理解的格式输出。


## 开发环境

- python 3.12
- mcp sdk
- uv
- pypdf
- paddle ocr


## 使用方法

```json
{
    "mcpServers":{
        "urlParseContent":{
            "command": "uv",
            "args":[
                "--directory",
                "xxx/urlParseContent",
                "run",
                "urlParseContent.py"
            ]
        }
    }
}
```
