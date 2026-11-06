# 맥에서 실습하기

이 문서는 **맥(Mac)으로 실습하는 분**을 위한 설치 안내입니다. 윈도우를 쓰신다면 `README.md`를 따르세요.  
과정 소개, 시간표, 수료 평가, 저작권 안내는 `README.md`에 있습니다.

맥에서는 아래 파일을 씁니다. 맥 GPU에서 생기는 오류를 피하고, 맥의 칩에 맞춰 실행되도록 고친 파일입니다.

| 용도 | 윈도우·리눅스 | 맥 |
|------|---------------|-----|
| 설치 파일 | `requirements.txt` | `requirements-mac.txt` |
| 1일차 | `1일차.ipynb` | `1일차_맥버전.ipynb` |
| 2일차 | `2일차.ipynb` 또는 `2일차_CPU버전.ipynb` | `2일차_맥버전.ipynb` |
| 최종과제 | `최종과제.ipynb` | `최종과제_맥버전.ipynb` |

---

## 1. 내 맥의 칩 확인하기

맥은 칩에 따라 **Apple Silicon 맥**과 **인텔 맥**으로 나뉘고, 설치할 Anaconda가 다릅니다.

1. 화면 왼쪽 위의 Apple 메뉴()를 누르고 **이 Mac에 관하여**를 엽니다.
2. 표시된 항목을 확인합니다.
    - **칩**: Apple M1, M2, M3, M4 등이 보이면 **Apple Silicon 맥**입니다.
    - **프로세서**: Intel Core i5, i7 등이 보이면 **인텔 맥**입니다.

| 구분 | Apple Silicon 맥 | 인텔 맥 |
|------|------------------|---------|
| 계산 장치 | 맥 GPU를 자동으로 사용 | CPU만 사용 |
| macOS 버전 | 12.3(Monterey) 이상 | 12(Monterey) 이상 |
| 메모리(RAM) | 16GB 이상 권장 | 16GB 이상 권장 |
| 2일차 LoRA 학습 시간 | 9분 정도 (M3 기준) | 1시간을 넘길 수 있음 |
| 최종과제 전체 실행 시간 | 20분 정도 (M3 기준) | Apple Silicon보다 훨씬 오래 걸림 |
| 2일차 7장(QLoRA) | 자동으로 건너뜀 | 자동으로 건너뜀 |
| 디스크 | 모델·패키지 포함 약 10GB 여유 공간 | 같음 |

노트북에 나오는 **MPS**는 PyTorch가 맥 GPU를 쓰는 기능의 이름입니다. 노트북은 칩을 스스로 확인해서 Apple Silicon 맥이면 `mps`, 인텔 맥이면 `cpu`로 실행합니다.  
QLoRA 실습은 NVIDIA GPU가 있을 때만 실행되도록 만들어져 있어서, 맥에서는 자동으로 건너뜁니다. 4비트 양자화 도구(`bitsandbytes`)도 맥 지원이 아직 제한적이라 맥용 설치 파일에서 뺐습니다.

---

## 2. Anaconda 설치

**Apple Silicon 맥**은 공식 다운로드 페이지에서 macOS용 설치 파일을 받아 설치합니다.  
https://www.anaconda.com/download/success

**인텔 맥**용 Anaconda 설치 파일은 2025년 6월 판(2025.06-1)이 마지막입니다. 공식 다운로드 페이지에서 인텔용을 찾을 수 없으면 아래 주소에서 받아 설치합니다.  
https://repo.anaconda.com/archive/Anaconda3-2025.06-1-MacOSX-x86_64.pkg

> **주의**: Apple Silicon 맥에 인텔용 Anaconda를 설치하지 마세요. 설치와 실행은 되지만 맥 GPU를 쓰지 못해 아주 느립니다.  
> 노트북의 첫 코드 셀이 `주의: Apple Silicon 맥에서 인텔용 파이썬을 쓰고 있어…`라고 알려 주면 이 경우이므로, Apple Silicon용 Anaconda로 다시 설치하세요.

---

## 3. 터미널에서 실습 폴더로 이동

맥에서는 **터미널** 앱에 명령을 입력합니다. `⌘ + Space`를 눌러 "터미널"을 검색하면 열 수 있습니다.

터미널에 `cd `(cd와 띄어쓰기 한 칸)를 입력한 뒤, Finder에서 실습 폴더를 터미널 창으로 끌어다 놓고 `Enter`를 누르면 그 폴더로 이동합니다.

---

## 4. Conda 가상환경 만들기 (Python 3.10)

```bash
conda create -n transformer_peft python=3.10
conda activate transformer_peft
```

- 처음 실행하면 Anaconda 이용 약관에 동의하라는 질문이 나올 수 있습니다. 안내에 따라 동의(accept)합니다.
- 설치를 계속할지 물으면 `y`를 입력합니다.

---

## 5. 실습에 필요한 패키지 설치

```bash
pip install -r requirements-mac.txt
```

이 파일 하나로 Apple Silicon 맥과 인텔 맥 모두 설치됩니다. 칩에 맞는 PyTorch 버전도 자동으로 골라집니다(Apple Silicon 2.5.1, 인텔 2.2.2).  
`transformers`, `peft`, `trl`은 버전에 따라 인자 이름이 달라지므로 **지정된 버전을 그대로 설치**하세요.

설치가 끝나면 아래 명령으로 확인할 수 있습니다.

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Apple Silicon 맥에서 `2.5.1 True`가 나오면 맥 GPU를 쓸 준비가 된 것입니다. 인텔 맥에서는 앞 숫자가 `2.2.2`이면 됩니다.

---

## 6. VS Code에서 열기

1. VS Code를 설치합니다. https://code.visualstudio.com/
2. VS Code를 실행하고 **파일 > 폴더 열기**에서 실습 폴더를 엽니다.
3. 노트북을 열고 오른쪽 위의 **커널 선택**에서 `transformer_peft` 환경을 고릅니다. Python, Jupyter 확장을 설치하라는 안내가 나오면 설치합니다.

---

## 7. 맥에서 실행할 때 알아 둘 점

- 1일차, 2일차, 최종과제 모두 **이름에 `_맥버전`이 붙은 노트북**을 여세요. 윈도우용 노트북은 맥에서 커널이 멈추거나 실행되지 않을 수 있습니다.
- **노트북을 여러 개 동시에 실행하지 마세요.** 노트북 하나가 메모리를 많이 씁니다. 메모리가 16GB인 Apple Silicon 맥(M3)에서 재 보니, 학습하는 동안 활동 모니터 기준으로 14~17GB까지 올라갔습니다. 다른 노트북을 열어 두었다면 그 노트북의 커널을 종료한 뒤 실행합니다.
- **실행하는 동안 맥 덮개를 닫지 마세요.** 맥이 잠자기에 들어가면 학습도 멈춥니다.
- 처음 실행할 때 Qwen 모델(약 1GB)을 내려받으므로 인터넷에 연결되어 있어야 합니다.
