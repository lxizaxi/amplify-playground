from datetime import timedelta
from os import environ
from typing import Final

import amplify
import dotenv
import numpy as np
from amplify import FixstarsClient, Poly, VariableGenerator
from util.question_util import QuestionUtil
from util.show_util import ShowUtil

dotenv.load_dotenv(override=True)

FIXSTARS_TOKEN: Final[str] = environ.get("FIXSTARS_TOKEN", "")

NUM_CITIES: Final[int] = 32
locations, distances = QuestionUtil.gen_random_tsp(NUM_CITIES)

# 1. 変数配列の作成
gen = VariableGenerator()
q = gen.array("Binary", shape=(NUM_CITIES + 1, NUM_CITIES))
q[NUM_CITIES, :] = q[0, :]

# 2. 目的関数の作成
objective: Poly = amplify.einsum("ij,ni,nj->", distances, q[:-1], q[1:])  # type: ignore

# 3. 制約条件
row_constraints = amplify.one_hot(q[:-1], axis=1)
col_constraints = amplify.one_hot(q[:-1], axis=0)
constrains = row_constraints + col_constraints

# 3'. 論文を参考に制約条件を修正
constrains *= np.amax(distances)
model = objective + constrains


# 4. ソルバークライアントの作成
client = FixstarsClient()
client.token = FIXSTARS_TOKEN
client.parameters.timeout = timedelta(milliseconds=1000)

# 4. ソルバーの実行
result = amplify.solve(model=model, client=client)

# 5. 結果の表示
print(f"{result.best.objective=}")
q_values = q.evaluate(result.best.values)
print(f"{q} = {q_values}")

# 図のplot
ShowUtil.show_plot(locations)
ShowUtil.show_route(
    route=np.where(np.array(q_values) == 1)[1],
    distances=distances,
    locations=locations,
    num_cities=NUM_CITIES,
)
