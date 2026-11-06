# 2026년 전공분야 보수교육(융합) [IT직종]
## **트랜스포머 이해와 LLM 효율적 파인튜닝(PEFT) 실습**

본 폴더는 **민간직업훈련기관 정보기술분야 교·강사 대상 보수교육(융합) 과정**  
「트랜스포머 이해와 LLM 효율적 파인튜닝(PEFT) 실습」의 교육자료 및 실습 예제 코드를 제공합니다.  
Transformer의 구조를 그림과 예제로 이해하고, Decoder-only LLM의 학습 방식을 살펴본 뒤, 공개 LLM(Qwen2.5)을 **SFT 데이터 구성 → LoRA·QLoRA 파인튜닝 → 평가**까지 적은 자원으로 조정하는 **실전형 파인튜닝 흐름**을 학습하는 것을 목표로 합니다.

---

## 교육 개요

| 구분 | 내용 |
|------|------|
| **교육구분** | 훈련 교·강사 보수교육(융합) |
| **교육대상** | 민간직업훈련기관 종사자 (NCS 중분류 2001 정보기술분야 교·강사) |
| **교육목표** | - RNN·Seq2Seq의 한계와 Transformer의 등장 배경 이해<br>- 입력 표현, Self-Attention, Multi-Head Attention, Encoder·Decoder 구조의 동작 원리 이해<br>- Decoder-only LLM의 사전학습·SFT·선호도 정렬 과정과 Prompt·RAG·Fine-tuning의 차이 이해<br>- SFT 데이터 구성, Chat Template, LoRA·QLoRA를 활용한 LLM 파인튜닝과 평가 실습 |
| **교육수준** | 중급 |
| **선수지식** | Python 기본 문법, 딥러닝 기초 개념(신경망, 손실 함수, 학습) |
| **교육시간** | 총 12시간 |
| **교육방법** | 집체 (이론 + 실습 중심) |
| **평가방법** | 실습평가 (LoRA 파인튜닝 과제 결과 제출) |

---

## 주요 습득 역량

- Transformer 기반 언어 모델의 구조 이해와 구현  
- 대규모 언어 모델(LLM)의 학습 단계와 업무 적용 방법(Prompt, RAG, Fine-tuning) 선택  
- SFT 학습 데이터 설계·정제·변환  
- PEFT(LoRA, QLoRA)를 활용한 LLM 파인튜닝과 성능 평가

---

## 교육 내용 요약

| 단원명 | 학습시간 | 학습방법 | 주요 내용 |
|---------|-----------|------------|-------------|
| Transformer 개요 | 1 | 이론 | RNN과 Seq2Seq의 한계, Transformer의 등장 배경과 발전 과정 |
| Transformer 구조 | 3 | 이론/실습 | 입력 처리(토큰화, 임베딩, 위치 정보), Self-Attention, Multi-Head, Encoder와 Decoder 구조 |
| LLM 구조와 학습 | 2 | 이론 | Decoder-only LLM, 사전학습·SFT·선호도 정렬, Prompt·RAG·Fine-tuning 비교 |
| SFT 데이터 구성 | 2 | 이론/실습 | 학습 데이터 품질 점검과 정제, 대화 형식 변환, Chat Template과 토큰화 |
| PEFT 이해 | 1 | 이론/실습 | Full Fine-tuning, LoRA, QLoRA 등 비교, PEFT 라이브러리로 LoRA 적용 |
| PEFT 파인튜닝 | 2 | 실습 | LLM 파인튜닝 환경 구성, `SFTTrainer` 기반 LoRA 학습, 어댑터 저장·병합 |
| 모델 평가 및 활용 | 1 | 실습/평가 | 학습 전후 정량 평가, 교육 내용 종합평가 |

---

## 세부 시간표 (수업 진행 상황에 따라 바뀔 수 있음)

| 일차 | 시간 | 주요 단원 | 학습 내용 | 방법 |
|------|------|------------|------------|------|
| 1일차 | 10:00–11:00 | Transformer 개요 | RNN과 Seq2Seq의 한계, Transformer의 등장 배경과 전체 구조 | 토론 |
|  | 11:00–12:00 | Transformer 입력 표현 | Tokenization, Embedding, 위치 정보(Positional Encoding, RoPE) | 이론/실습 |
|  | 13:00–14:00 | Attention과 Encoder | Self-Attention, Q·K·V, Multi-Head, Encoder 블록 구성 | 이론/실습 |
|  | 14:00–15:00 | Transformer Decoder | Masked Attention, Cross-Attention, 학습과 추론의 차이 | 이론/실습 |
|  | 15:00–16:00 | Decoder-only LLM | Transformer에서 GPT와 생성형 LLM으로의 발전, Qwen2.5 구조 분석 | 이론/실습 |
|  | 16:00–17:00 | LLM 학습과 활용 | 사전학습, SFT, 선호도 정렬 / Prompt, RAG, Fine-tuning 비교 | 이론/문제풀이 |
| 2일차 | 09:00–10:00 | 실습 환경 구성 | Hugging Face 실습 환경 설정, Qwen2.5 모델 실행, GPU 메모리 계산 | 이론/실습 |
|  | 10:00–11:00 | SFT 데이터 구성 | 학습 데이터 품질 확인, 정제, 대화형 데이터 변환과 분리 | 이론/실습 |
|  | 11:00–12:00 | Chat Template와 Tokenization | 대화 형식의 토큰화, input_ids·labels 구조, 답변 토큰만 학습 | 이론/실습 |
|  | 13:00–14:00 | PEFT와 LoRA | Full Fine-tuning과 PEFT 비교, LoRA·QLoRA 원리와 구현 | 이론/실습 |
|  | 14:00–15:00 | LoRA 모델 학습 | LoRA 설정과 `SFTTrainer` 기반 파인튜닝, 평가와 활용 | 실습 |
|  | 15:00–16:00 | 종합평가 | 고객 문의 응답 형식 LoRA 파인튜닝 과제 | 실습/평가 |

---

## 실습 환경 준비

본 과정의 모든 실습은 **Python 3.10 기반 Conda 가상환경**에서 진행됩니다.  
아래 절차를 완료하면 VS Code에서 노트북을 실행할 수 있습니다.

### 권장 하드웨어

| 구분 | 1일차 | 2일차 · 최종과제 |
|------|-------|------------------|
| GPU | 필요 없음 (CPU로 실행 가능) | **NVIDIA GPU (VRAM 8GB 이상) 권장** |
| CPU만 있는 경우 | 실행 가능 (번역 학습 시간은 길어질 수 있음) | LoRA 학습 가능하나 시간이 크게 늘어날 수 있음, QLoRA 실습은 건너뜀 |
| 메모리(RAM) | 8GB 이상 | 16GB 이상 권장 |
| 디스크 | 모델·패키지 포함 약 10GB 여유 공간 | |

GPU가 없는 환경에서는 Google Colab(T4 GPU)에서 노트북을 열어 실행해도 됩니다. 이 경우 `requirements.txt`의 패키지를 설치하면 바로 실행할 수 있습니다.

### 1. Anaconda 설치

운영체제에 맞는 Anaconda를 먼저 설치합니다.

Anaconda 공식 다운로드 페이지:  
https://www.anaconda.com/download/success

### 2. Conda 가상환경 생성 (Python 3.10)

```bash
conda create -n transformer_peft python=3.10
```

### 3. 가상환경 활성화

```bash
conda activate transformer_peft
```

### 4. 실습에 필요한 패키지 설치

본 저장소 루트 디렉터리에 포함된 `requirements.txt` 파일을 사용하여  
실습에 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

`requirements.txt`는 NVIDIA GPU(CUDA 12.1)용 PyTorch를 설치하도록 설정되어 있습니다.  
실습 코드는 `transformers`, `peft`, `trl`의 버전에 따라 인자 이름이 달라질 수 있으므로 **지정된 버전을 그대로 설치**하는 것을 권장합니다.

### 5. VS Code 설치

본 과정의 실습은 VS Code에서 노트북을 열어 진행합니다.  
VS Code가 설치되어 있지 않은 경우 아래 공식 페이지에서 설치합니다.

https://code.visualstudio.com/

### 6. VS Code에서 열기

```bash
code .
```

VS Code에서 노트북을 열고 커널을 `transformer_peft` 환경으로 선택합니다.

---

## 수료 평가 안내

수료 평가는 `최종과제.ipynb`의 **고객 문의 응답 형식 LoRA 파인튜닝** 실행 결과를 정리한 캡처 보고서로 진행합니다.

제출물은 **캡처 보고서 1개**입니다.

- 파일명: `수행평가 제출양식_본인이름.docx`  
    - 본인 이름으로 변경해서 제출합니다.  
    - 예시: `수행평가 제출양식_김민수.docx`  
- 보고서 분량: 1~3쪽  
- 실행한 노트북 파일은 선택 제출입니다.

보고서에는 아래 실행 결과를 포함합니다.

| 번호 | 필수 캡처 | 노트북 위치 |
|---|---|---|
| 1 | 학습 전 평가 결과 (문의만 / 형식 지시 프롬프트) | `3. 학습 전 평가` |
| 2 | 학습 파라미터 비율과 손실 기록 | `4. LoRA 설정과 학습` |
| 3 | 학습 전후 비교표 | `5. 학습 후 평가` |
| 4 | 설정을 바꿔 다시 학습한 결과 | `6. 설정 바꿔 다시 학습하기` |

총점은 100점이며, 수료 기준은 60점입니다.

| 항목 | 점수 | 채점 기준 |
|---|---:|---|
| 제출 형식 | 20점 | 지정된 파일명 형식의 DOCX 보고서 제출 |
| 학습 전 평가 | 20점 | 문의만 넣은 경우와 형식 지시 프롬프트를 넣은 경우의 결과 포함 |
| LoRA 학습 | 20점 | 학습 가능한 파라미터 비율과 epoch별 손실 기록 포함 |
| 학습 후 평가 | 20점 | 학습 전후 비교표 포함, 학습 후 형식 준수율 80% 이상 |
| 설정 비교 및 설명 | 20점 | 바꾼 설정과 지표 변화, 네 가지 설명 항목 작성 |

---

## 저작권 및 문의

본 교육자료의 **무단 복제, 배포, 상업적 이용을 금지합니다.**  
교육자료의 사용 허가나 기타 문의사항은 별도 관리 지침에 따릅니다.

실습에 사용하는 Qwen2.5 모델은 [Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)을 따릅니다.

---

> © 2026. 트랜스포머 이해와 LLM 효율적 파인튜닝(PEFT) 실습  
> 본 자료는 훈련 교·강사 보수교육(융합) 과정 전용으로 제작되었습니다.  
> All rights reserved.
