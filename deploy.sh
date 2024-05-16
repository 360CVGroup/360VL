# pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
# pip config set global.trusted-host mirrors.cloud.tencent.com
pip install --upgrade pip  # enable PEP 660 support

pip install -e .

pip install ninja
pip install flash-attn --no-build-isolation
