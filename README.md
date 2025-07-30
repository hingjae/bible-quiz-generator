# 📘 Bible Quiz Generator

RAG(Retrieval-Augmented Generation) 기반 성경 퀴즈 생성기입니다.  
LangChain, OpenAI, Pinecone 벡터스토어를 활용해 특정 주제나 무작위 성경 내용을 기반으로 객관식 4지선다형 문제를 자동으로 생성합니다.

---

## 가상환경 실행

```bash
python -m venv venv
source venv/bin/activate
```

## 벡터라이징

```bash
python ./vector/vector.py
```

## 로컬 lambda 실행 방법

```bash
docker compose up -d --build
```

## 로컬 lambda 테스트 예시

```curl
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"bookTitle": "Genesis", "question": "요셉"}'
```

## lambda layer 압축

```bash
make build-zip
```
