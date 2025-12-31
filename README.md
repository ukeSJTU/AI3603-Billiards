# AI3603-Billiards

AI3603 课程台球大作业

---

loguru logging 功能：

-   CRITICAL：用来记录当前程序运行到这里，无论如何都无法恢复的情况
-   ERROR：程序出错，但是我们提供了兜底的机制，例如 try 或者重试等等，功能失败但程序继续
-   WARNING：出现了不应该出现的情况，但是仍然继续执行
-   INFO：程序正常运行时，记录一些重要的事件（直接在 console 输出的信息）
-   DEBUG：调试信息，最全的日志信息，等待程序运行完成后再查看 log 文件

---

项目使用了`uv`来管理环境，在安装[`pooltool`]()这个依赖的过程中，遇到了比较麻烦的问题，解决办法是：

```bash
uv add pooltool-billiards --index https://archive.panda3d.org/ --index-strategy unsafe-best-match --prerelease allow
```

我个人认为问题在于`pooltool`这个包依赖的`panda3d`版本问题，可以看到[](https://pooltool.readthedocs.io/en/latest/getting_started/install.html)中推荐的安装方式是：

```bash
pip install pooltool-billiards --extra-index-url https://archive.panda3d.org/
```

> (Providing the Panda3D archive is required until Panda3D v1.11 is released)

TODO: 这里需要进一步补充信息

总之配置完成后，运行：

```bash
python -c "import pooltool; print(pooltool.__version__)"
0.5.0
```

---

Run code below to install pre-commit hooks:

```bash
uv run pre-commit install
```
