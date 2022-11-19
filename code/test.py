import requests
import configparser
import json

config_ini = configparser.ConfigParser()
config_ini.read("../../datasets/config/config (1).ini", encoding="utf-8")
WANDB_API_KEY = config_ini["WANDB_API"]["API_KEY"]
NOTION_API_KEY = "secret_mDRpQpcMtlPJiaLh4jG7MHDzJK6iAAjQixkEwVwK3yC"

print(NOTION_API_KEY)

url = f"https://api.notion.com/v1/pages"
database_id = "93d0f104b15e4dab83b24b4aa26fe563"

headers = {"Authorization": f"Bearer {NOTION_API_KEY}",
           "Content-Type": "application/json",
           "Notion-Version": "2022-06-28"
          }
body = {
    "parent": {
        "database_id": database_id
    },
    "properties": {
        "Name": {"title": [{"text": {"content": str("test")}}]},
        "model": {"rich_text":[{"text": {"content": str("test")}}]},
        "max_len": {"rich_text":[{"text": {"content": str("test")}}]},
        "fold": {"rich_text":[{"text": {"content": str("test")}}]},
        "eps": {"rich_text":[{"text": {"content": str("test")}}]},
        "scheduler": {"rich_text":[{"text": {"content": str("test")}}]},
        "batch_size": {"rich_text":[{"text": {"content": str("test")}}]},
        "coments": {"rich_text":[{"text": {"content": str("test")}}]},
        "score": {"rich_text":[{"text": {"content": str("test")}}]},
    }
}
response = requests.request('POST', url=url, headers=headers, data=json.dumps(body))
print(response)
print(headers)