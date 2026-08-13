import asyncio
import aiohttp
import nest_asyncio
import json
import time
import ssl
import aiofiles
import os

data_type = "item"
data_name = "mooc"

input_path = f"../data/{data_name}/llm_input_{data_type}.json"
write_path = f"../data/{data_name}/llm_response_{data_type}.json"

system_input = ""

if data_type == "item":
    if data_name == "ml-1m":
        system_input = "Assume you are an expert in movie recommendation. You will be given a certain movie with its first-order information (in the form of triples) and some second-order relationships (movies related to this movie). Please complete the missing knowledge, summarize the movie and analyze what kind of users would like it. Your response should be a coherent paragraph and no more than 200 words."
    elif data_name == "mooc":
        system_input = (
            "假设你是一名在线教育课程推荐专家。你将获得一门目标课程的一阶信息，可能包括课程名称、课程简介、所属领域、开课学校、授课教师、先修要求和核心概念；同时还会获得从知识图谱中采样的二阶关联信息，例如与目标课程共享概念、领域、教师或学校的其他课程。请综合这些信息，生成一段连贯、客观的课程描述，概括课程的主要内容、知识领域、先修基础、可能的学习难度、能够培养的能力，以及适合学习该课程的用户类型。只能使用输入中能够支持的信息，不要虚构教师经历、课程考核方式、课程质量、学习效果或未提供的事实。输出应为一个连贯的中文段落，不超过300个汉字。"
        )
elif data_type == "user":
    if data_name == "ml-1m":
        system_input = "Assume you are an expert in movie recommendation with access to a viewer's movie-watching history, where each entry is formatted as (movie_name: genres: xx, director: xx, main actors: xx, overview: xx). Please analyze and summarize this user's viewing preferences from the aspects of movie genres, directors, and actors. Your response should be a coherent and fluent paragraph, not exceeding 100 words."
    elif data_name == "mooc":
        system_input = (
            "假设你是一名在线教育课程推荐专家。你将获得一名用户在训练集中学习过的课程历史。每条历史记录可能包含课程名称、所属领域、授课教师、开课学校、先修要求、课程简介和核心概念。请根据这些课程记录，总结该用户的学习兴趣、偏好的知识领域、经常接触的课程主题、可能具备的知识基础，以及适合继续推荐的课程类型。不要根据课程历史推断用户的年龄、性别、职业、学校、真实能力水平或其他未提供的个人属性；不要推荐具体的测试集课程名称。输出应为一个连贯、客观的中文段落，不超过150个汉字。"
        )

# 读取输入
with open(input_path, "r", encoding="utf-8") as f:
    input_dic = json.load(f)

requests = [
    {"role": "user", "content": request, "item_key": key}
    for key, request in input_dic.items()
]

# ========================
# ✅ 核心修复参数
# ========================
BASE_URL = ""  # ✔ 修复404关键点

concurrent_limit = 3          # ✔ 控制并发
request_interval = 0.8        # ✔ 控制QPS
batch_size = 10               # ✔ 防止瞬时爆发
max_retries = 6
batch_delay = 3

ssl_context = ssl.create_default_context()
semaphore = asyncio.Semaphore(concurrent_limit)

api_key = ""

# ========================
# 请求函数（稳定版）
# ========================
async def fetch_response(session, request, retries=0):

    async with semaphore:
        await asyncio.sleep(request_interval)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "qwen-max",
            "messages": [
                {"role": "system", "content": system_input},
                {
                    "role": "user",
                    "content": request["content"]
                }
            ],
            "temperature": 0.0,
            "top_p": 0.001,
            "stream": False
        }

        try:
            async with session.post(
                BASE_URL,
                headers=headers,
                data=json.dumps(payload),
                ssl=ssl_context,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

                elif response.status == 429:
                    wait = min(2 ** retries, 60)
                    print(f"429 rate limit, retry {retries+1}, sleep {wait}s")
                    await asyncio.sleep(wait)

                    if retries < max_retries:
                        return await fetch_response(session, request, retries + 1)
                    return None

                elif response.status == 404:
                    text = await response.text()
                    print(f"404 ERROR (URL or model issue): {text}")
                    return None

                else:
                    text = await response.text()
                    print(f"HTTP {response.status}: {text}")
                    return None

        except Exception as e:
            if retries < max_retries:
                wait = 2 ** retries
                print(f"Exception retry {retries+1}: {e}, sleep {wait}s")
                await asyncio.sleep(wait)
                return await fetch_response(session, request, retries + 1)
            return None


# ========================
# batch 控制（关键修复）
# ========================
async def process_batch(session, batch):
    results = []

    # ✔ 改掉 gather（避免瞬时并发爆炸）
    for req in batch:
        res = await fetch_response(session, req)
        results.append(res)

    return results


async def write_responses_to_file(responses, file_path):
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(responses, ensure_ascii=False, indent=2))


async def main(file_path):

    responses = {}

    connector = aiohttp.TCPConnector(
        limit=concurrent_limit,
        ssl=ssl_context
    )

    async with aiohttp.ClientSession(connector=connector) as session:

        for i in range(0, len(requests), batch_size):

            print(f"\nBatch {i // batch_size + 1}")

            batch = requests[i:i + batch_size]

            batch_responses = await process_batch(session, batch)

            for req, resp in zip(batch, batch_responses):
                responses[req["item_key"]] = resp

            await write_responses_to_file(responses, file_path)

            print(f"Batch done, sleep {batch_delay}s")
            await asyncio.sleep(batch_delay)

    return responses


# ========================
# run
# ========================
if os.path.exists(write_path):
    os.remove(write_path)

responses = asyncio.run(main(write_path))

print("DONE:", len(responses))