from datetime import timedelta
from os import environ
from typing import Final

import amplify
import dotenv
from amplify import FixstarsClient, VariableGenerator

dotenv.load_dotenv(override=True)

FIXSTARS_TOKEN: Final[str] = environ.get("FIXSTARS_TOKEN", "")


# 1. 変数配列の作成
gen = VariableGenerator()
q = gen.array("Binary", 2)

# 2. 目的関数の作成
f = q[0] * q[1] + q[0] - q[1] + 1

# 3. ソルバークライアントの作成
client = FixstarsClient()
client.token = FIXSTARS_TOKEN
client.parameters.timeout = timedelta(milliseconds=1000)

# 4. ソルバーの実行
result = amplify.solve(model=f, client=client)

# 5. 結果の表示
print(f"{result.best.objective=}")
print(f"{q} = {q.evaluate(result.best.values)}")
