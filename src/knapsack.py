from dataclasses import dataclass
from datetime import timedelta
from os import environ
from typing import Final

import amplify
import dotenv
import numpy as np
from amplify import Constraint, FixstarsClient, Poly, VariableGenerator

dotenv.load_dotenv(override=True)

FIXSTARS_TOKEN: Final[str] = environ.get("FIXSTARS_TOKEN", "")


@dataclass
class Jewel:
    value: int
    weight: int


jewels: list[Jewel] = [
    Jewel(95000, 470),
    Jewel(60000, 300),
    Jewel(25000, 130),
    Jewel(12000, 70),
    Jewel(1000, 30),
]

capacity = 500

# 1. 変数配列の作成
gen = VariableGenerator()
q = gen.array("Binary", len(jewels))

objective: Poly = amplify.einsum("i,i->", np.array([j.value for j in jewels]), q)  # type: ignore

constraint: Constraint = amplify.less_equal(
    amplify.einsum("i,i->", np.array([j.weight for j in jewels]), q), capacity
)  # type: ignore

# 問題を定義
problem = -1 * objective + 1e5 * constraint.penalty

# 3. ソルバークライアントの作成
client = FixstarsClient()
client.token = FIXSTARS_TOKEN
client.parameters.timeout = timedelta(milliseconds=1000)

# 4. ソルバーの実行
result = amplify.solve(model=problem, client=client)

# 5. 結果の表示
print(f"{result.best.objective=}")
print(f"{q} = {q.evaluate(result.best.values)}")
