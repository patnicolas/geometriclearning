__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
