import argparse


def main():
    parser = argparse.ArgumentParser(description="경구약제 객체 탐지 학습")
    parser.add_argument("--config", default="configs/default.yaml", help="config 파일 경로")
    args = parser.parse_args()
    print(f"config: {args.config}")
    # TODO: config 로드 → 모델 빌드 → 학습 루프


if __name__ == "__main__":
    main()
