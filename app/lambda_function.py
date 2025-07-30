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

load_dotenv()

def send_to_sqs(message_body):
  sqs = boto3.client(
      'sqs',
      region_name=os.getenv("AWS_REGION"),
      endpoint_url=os.getenv("SQS_ENDPOINT"),
      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
  )

  # 큐 URL 가져오기
  queue_name = os.getenv("QUEUE_NAME")
  queue_url = sqs.get_queue_url(QueueName=queue_name)['QueueUrl']

  # 메시지 전송
  response = sqs.send_message(
      QueueUrl=queue_url,
      MessageBody=json.dumps(message_body)
  )
  print("SQS 전송 결과:", response)

def lambda_handler(event, context):

  # topic_id=event["topic_id"]
  book_title = event.get("bookTitle")
  question = event.get("question")

  QUIZ_MULTI_PROMPT_TEMPLATE = """
  당신은 성경 전문가입니다.
  아래 **Context** 내용을 바탕으로 객관식 4지선다형 퀴즈를 총 10개를 한글로 생성하세요.

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

  def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

  PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
  PINECONE_INDEX = os.getenv("PINECONE_INDEX")

  pc = Pinecone(api_key=PINECONE_API_KEY)

  index = pc.Index(PINECONE_INDEX)

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

  vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=book_title)

  retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.4}
  )

  llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

  prompt_template = PromptTemplate(
    input_variables=["context"],
    template=QUIZ_MULTI_PROMPT_TEMPLATE
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
    
    random_numbers = random.sample(
        range(total_vector_count),
        k=5
    )

    random_ids =[f"{book_title}-{i}" for i in random_numbers]
    print(f"random_ids : {random_ids}")

    response = index.fetch(namespace=book_title, ids=random_ids)

    docs = [
        Document(page_content=vector_data.metadata["text"])
        for vector_data in response.vectors.values()
    ]

    context = "\n\n".join([doc.page_content for doc in docs])

    filled_prompt = prompt_template.format(context=context)
    response = llm.invoke(filled_prompt)

  cleaned = response.content.strip()
  cleaned = cleaned.replace("```json", "").replace("```", "").strip()

  parsed = json.loads(cleaned)

  send_to_sqs(parsed)

  return {
    "statusCode": 200,
    "body": parsed
  }
