from causalens.tools.calculator import calculator
from causalens.tools.data_profile import profile_dataset


def main() -> None:
    print("=== Calculator: 成功示例 ===")
    print(calculator("1 + 2 * 3").model_dump())

    print("\n=== Calculator: 失败示例 ===")
    print(calculator("hello").model_dump())

    print("\n=== Data Profile: 成功示例 ===")
    print(profile_dataset("data/sample.csv").model_dump())

    print("\n=== Data Profile: 失败示例 ===")
    print(profile_dataset("data/not_found.csv").model_dump())


if __name__ == "__main__":
    main()