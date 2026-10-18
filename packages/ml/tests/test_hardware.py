from padel_ml.thesis.hardware import GIB, cpu_model, whole_gb


def test_the_core_count_linux_appends_is_dropped() -> None:
    assert cpu_model("AMD Ryzen 9 5900X 12-Core Processor") == "AMD Ryzen 9 5900X"


def test_a_name_without_core_count_is_kept() -> None:
    assert cpu_model("Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz ") == (
        "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz"
    )


def test_memory_is_given_in_whole_gigabytes() -> None:
    assert whole_gb(34359738368, 1) == 32
    assert whole_gb(24576, 1024**2) == 24
    assert whole_gb(GIB, 1) == 1
