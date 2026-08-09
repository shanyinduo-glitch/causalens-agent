from causalens.tools.data_profile import profile_dataset


def main() -> None:
    print("=== Data Profile: 成功示例 ===")
    print(
        profile_dataset(
            "data/sample.csv"
        ).model_dump()
    )

    print("\n=== Data Profile: 失败示例 ===")
    print(
        profile_dataset(
            "data/not_found.csv"
        ).model_dump()
    )


if __name__ == "__main__":
    main()