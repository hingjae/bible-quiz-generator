from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document

import json
import random
import os
import boto3
import uuid
import traceback

load_dotenv()

# =========================
# 환경 변수
# =========================
PROFILE            = os.getenv("PROFILE", "")
REGION             = os.getenv("REGION", "ap-northeast-2")
SQS_ENDPOINT       = os.getenv("SQS_ENDPOINT")  # 로컬에서만 필요
RESULT_QUEUE_URL   = os.getenv("RESULT_QUEUE_URL")  # 권장: 결과 큐 URL 직접 주입
RESULT_QUEUE_NAME  = os.getenv("QUEUE_NAME")        # (백업) 이름만 있는 경우 사용

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX")

# =========================
# 전역(콜드스타트 1회) 초기화: 외부 클라이언트/LLM
# =========================
pc         = Pinecone(api_key=PINECONE_API_KEY)
index      = pc.Index(PINECONE_INDEX)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
llm        = ChatOpenAI(model="gpt-4o", temperature=0.3)

QUIZ_MULTI_PROMPT_TEMPLATE = """
당신은 성경 전문가입니다.
아래 **Context** 내용을 바탕으로 객관식 4지선다형 퀴즈를 총 3개를 한글로 생성하세요.

**출제 규칙:**
- 반드시 아래 JSON 배열 형식으로만 반환하세요.
- 절대 코드블럭(````json`)이나 추가 설명을 포함하지 마세요.
- 질문(question), 보기(options), 정답(correct), 정답의 이유(correct_answer_reason)는 모두 **한글로** 작성하세요.
- 정답(correct_answer)는 보기(options) 중 하나와 **정확히 일치하는 값으로 작성**하세요.
- 보기(options) 포함되지 않은 값을 정답(correct_answer)으로 작성하지 마세요.
- 정답의 이유(correct_answer_reason)는 성경 본문에 기반한 논리적인 해설이어야 합니다.
- 정답의 근거가 되는 성경 말씀의 출처를 "reference" 필드에 "창1:1", "창1:1-3", "출12:7,11" 처럼 **한 개 또는 복수의 장절**로 작성하세요.
- Context에 충분한 정보가 없다면 배열의 각 항목에 "충분한 정보가 없습니다."라는 문제와 함께 options는 빈 배열, reference는 빈 문자열("")로 반환하세요.

**질문(question) 작성 가이드라인:**
- 각 질문(question)은 반드시 서로 다른 내용을 기반으로 생성하세요.
- 사건의 순서, 인물 관계, 장소 등 **성경 본문에서 명확히 파악 가능한 내용을** 기반으로 작성하세요.
- 성경의 앞 뒤 맥락을 고려하여 작성하세요. 시점과 대상이 모호한 질문(question)은 절대 만들지 마세요.
- 질문(question)은 **자연스러운 한국어 문장 구조로**, 누가(주어), 무엇을(목적어), 언제/왜/어디서(부사어)를 명확하게 작성하세요.
- 질문(question)은 **'언제', '누가', '무엇을', '어디서', '왜'** 등을 포함해 구체적인 문장으로 작성하세요.
- 반드시 **하나의 정답(correct_answer)만 도출될 수 있도록** 질문(question)과 보기(options)를 구성하세요.
- 복수의 정답처럼 보일 수 있는 선택지(options)는 만들지 마세요.
- 시점이나 행위 주체가 불분명하거나, 해석이 나뉘는 본문은 출제하지 마세요.

**JSON 배열 형식:**
[
  {{
    "question": "하나님이 태초에 창조하신 것은 무엇입니까?",
    "options": ["하늘과 땅", "빛과 어둠", "사람", "바다와 육지"],
    "correct_answer": "하늘과 땅",
    "correct_answer_reason": "'태초에 하나님이 천지를 창조하시니라'는 창세기 1:1의 말씀에 근거합니다.",
    "reference": "창1:1"
  }},
  {{
    "question": "하나님이 아담을 위해 여자를 만드실 때 사용한 것은 무엇입니까?",
    "options": ["머리카락", "갈빗대", "흙", "심장"],
    "correct_answer": "갈빗대",
    "correct_answer_reason": "창세기 2:21-22에 따르면, 하나님은 아담의 갈빗대 하나를 취하여 여자를 만드셨습니다.",
    "reference": "창2:21-22"
  }}
  ...
]

**Context:**

{context}
"""
prompt_template = PromptTemplate(input_variables=["context"], template=QUIZ_MULTI_PROMPT_TEMPLATE)


# =========================
# 유틸
# =========================
def get_sqs_client():
    if PROFILE == "local" and SQS_ENDPOINT:
        return boto3.client(
            "sqs",
            region_name=REGION,
            endpoint_url=SQS_ENDPOINT,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    return boto3.client("sqs", region_name=REGION)


def resolve_result_queue_url(sqs):
    """
    권장: RESULT_QUEUE_URL을 환경변수로 직접 주입.
    (백업) RESULT_QUEUE_NAME만 있을 땐 GetQueueUrl 사용.
    """
    if RESULT_QUEUE_URL:
        return RESULT_QUEUE_URL
    if RESULT_QUEUE_NAME:
        return sqs.get_queue_url(QueueName=RESULT_QUEUE_NAME)["QueueUrl"]
    raise RuntimeError("RESULT_QUEUE_URL 또는 QUEUE_NAME 환경변수가 필요합니다.")


def send_to_sqs(parsed_response, request_id):
    sqs = get_sqs_client()
    queue_url = resolve_result_queue_url(sqs)

    message_body = {"request_id": request_id, "quizzes": parsed_response}
    resp = sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(message_body))
    print("SQS 전송 결과 MessageId:", resp.get("MessageId"))
    return request_id


def iter_payloads(event):
    """
    - SQS 트리거: event.Records[*].body(JSON 문자열) → dict로 변환
    - 직접 invoke: event 자체 사용
    """
    if "Records" in event and event["Records"]:
        for rec in event["Records"]:
            body = rec.get("body", "{}")
            try:
                obj = json.loads(body)
            except Exception:
                obj = {"raw_body": body}
            # 실패 응답에 쓰기 위해 레코드 정보 포함
            obj["_record"] = {"messageId": rec.get("messageId")}
            yield obj
    else:
        event["_record"] = None
        yield event


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def process_one(payload):
    """
    기존 RAG + LLM 로직을 함수로 분리.
    실패 시 예외를 던져 상위에서 partial failure 처리.
    """
    book_title = payload.get("bookTitle")
    question   = payload.get("question")

    if not book_title:
        raise ValueError("bookTitle이 필요합니다.")

    vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=book_title)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.4}
    )

    if question:
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
        )
        response = rag_chain.invoke(question)
    else:
        ids_response = index.describe_index_stats()
        total_vector_count = ids_response.total_vector_count
        random_numbers = random.sample(range(total_vector_count), k=min(5, total_vector_count))
        random_ids = [f"{book_title}-{i}" for i in random_numbers]
        print("random_ids:", random_ids)

        fetched = index.fetch(namespace=book_title, ids=random_ids)
        docs = [Document(page_content=v.metadata["text"]) for v in fetched.vectors.values()]
        context = "\n\n".join([d.page_content for d in docs])

        filled_prompt = prompt_template.format(context=context)
        response = llm.invoke(filled_prompt)

    cleaned = (response.content or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # JSON 파싱 실패 시 재시도/ DLQ 로 흐르게 예외 발생
    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        print("JSON parse error:", e)
        print("LLM raw output:", cleaned[:2000])
        raise

    return parsed


# =========================
# Lambda 핸들러
# =========================
def lambda_handler(event, context):
    """
    - SQS 트리거: 부분 배치 실패 응답을 사용(ReportBatchItemFailures)
    - 직접 invoke: 200 반환
    """
    is_sqs = "Records" in event
    failures = []

    for payload in iter_payloads(event):
        record = payload.pop("_record", None)
        # 상관관계 ID: 요청에서 들어온 requestId 유지(없으면 새로 생성)
        request_id = payload.get("requestId") or str(uuid.uuid4())

        try:
            parsed = process_one(payload)
            send_to_sqs(parsed, request_id)
        except Exception as e:
            print("ERROR processing payload:", request_id, e)
            traceback.print_exc()
            if is_sqs and record:
                failures.append({"itemIdentifier": record["messageId"]})
            else:
                # 직접 invoke인 경우엔 예외를 올려서 5xx로 처리
                raise

    if is_sqs:
        # ✅ 부분 배치 실패 응답: 실패한 messageId만 재시도
        return {"batchItemFailures": failures}
    else:
        # 직접 invoke일 때만 의미 있음
        return {"statusCode": 200, "requestId": str(uuid.uuid4())}
        