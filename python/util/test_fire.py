import fire


class TestFire(object):
    def __init__(self, count: int) -> None:
        self.count = count

    def __str__(self) -> str:
        return str(self.count)

    @staticmethod
    def show(count: int) -> int:
        return count

    @staticmethod
    def show2() -> str:
        return str(TestFire(11))


if __name__ == '__main__':
    fire.Fire(TestFire.show(4))
    fire.Fire(TestFire.show2())
