# 下面这个脚本用来比较 copy.deepcopy 和 pooltool 中对象本身提供的 copy 方法（内部是 attrs 提供的 evolve 方法）在复制复杂对象时的表现差异。

import copy
import sys
import time
from typing import Any, Callable, Set, Tuple

import numpy as np
import pooltool as pt
from memory_profiler import memory_usage

# ============================================================================
# Synthetic Object Generators
# ============================================================================


def create_simple_ball(ball_id: str = "cue") -> pt.Ball:
    """Create a simple ball with minimal history."""
    return pt.Ball.create(ball_id, xy=(0.5, 0.5))


def create_complex_ball(ball_id: str = "1", history_length: int = 100) -> pt.Ball:
    """Create a ball with extensive history by simulating multiple shots."""
    # Always use "cue" as the ball_id for proper system initialization
    ball = pt.Ball.create("cue", xy=(0.5, 0.5))

    # Create multiple simulations to build up history
    if history_length > 0:
        # Add another ball to avoid empty collision cache
        balls = {"cue": ball, "1": pt.Ball.create("1", xy=(1.2, 0.5))}

        for _ in range(min(history_length // 20, 5)):  # Do up to 5 simulations
            system = pt.System(table=pt.Table.default(), cue=pt.Cue.default(), balls=balls)

            # Give it some velocity to create history during simulation
            system.balls["cue"].state.rvw[1] = np.array(
                [np.random.uniform(0.3, 0.8), np.random.uniform(-0.2, 0.2), 0.0]
            )

            # Evolve the system to create history
            try:
                pt.simulate(system, continuous=True)
            except Exception:
                pass  # Ignore simulation errors, we just want some history

            ball = system.balls["cue"]
            balls = {"cue": ball, "1": pt.Ball.create("1", xy=(1.2, 0.5))}

    return ball


def create_simple_table() -> pt.Table:
    """Create a simple standard table."""
    return pt.Table.default()


def create_simple_system(num_balls: int = 2) -> pt.System:
    """Create a simple system with few balls and no history."""
    balls = {"cue": pt.Ball.create("cue", xy=(0.5, 0.5))}

    # Add numbered balls
    for i in range(1, num_balls):
        balls[str(i)] = pt.Ball.create(str(i), xy=(1.0 + i * 0.1, 0.5 + i * 0.1))

    system = pt.System(table=pt.Table.default(), cue=pt.Cue.default(), balls=balls)

    return system


def create_complex_system(num_balls: int = 16, history_length: int = 50) -> pt.System:
    """Create a complex system with many balls and histories."""
    balls = {"cue": pt.Ball.create("cue", xy=(0.5, 0.5))}

    # Add numbered balls
    for i in range(1, num_balls):
        balls[str(i)] = pt.Ball.create(str(i), xy=(1.0 + i * 0.1, 0.5 + i * 0.1))

    system = pt.System(table=pt.Table.default(), cue=pt.Cue.default(), balls=balls)

    # Simulate to create history if requested
    if history_length > 0:
        # Give cue ball velocity
        system.balls["cue"].state.rvw[1] = np.array([0.5, 0.3, 0.0])
        pt.simulate(system, continuous=True)

    return system


# ============================================================================
# Correctness Tests
# ============================================================================


def test_ball_copy_independence(ball: pt.Ball) -> Tuple[bool, bool]:
    """
    Test if copying a ball creates independent copies.
    Returns (deepcopy_independent, native_copy_independent)
    """
    # Test copy.deepcopy
    ball_deepcopy = copy.deepcopy(ball)
    original_pos = ball.state.rvw[0].copy()
    ball_deepcopy.state.rvw[0, 0] += 1.0  # Modify x position
    deepcopy_independent = np.allclose(ball.state.rvw[0], original_pos)

    # Test native copy
    ball_native = ball.copy()
    ball_native.state.rvw[0, 1] += 1.0  # Modify y position
    native_independent = np.allclose(ball.state.rvw[0], original_pos)

    return deepcopy_independent, native_independent


def test_table_copy_independence(table: pt.Table) -> Tuple[bool, bool]:
    """
    Test if copying a table creates independent copies.
    Returns (deepcopy_independent, native_copy_independent)
    """
    # Test copy.deepcopy
    table_deepcopy = copy.deepcopy(table)
    _original_pocket_count = len(table.pockets)
    if table_deepcopy.pockets:
        # Try to modify pocket contains set
        first_pocket = list(table_deepcopy.pockets.values())[0]
        first_pocket.contains.add("test_ball")
        deepcopy_independent = len(list(table.pockets.values())[0].contains) < len(
            first_pocket.contains
        )
    else:
        deepcopy_independent = True

    # Test native copy
    table_native = table.copy()
    if table_native.pockets:
        first_pocket = list(table_native.pockets.values())[0]
        first_pocket.contains.add("test_ball_2")
        native_independent = len(list(table.pockets.values())[0].contains) < len(
            first_pocket.contains
        )
    else:
        native_independent = True

    return deepcopy_independent, native_independent


def test_system_copy_independence(system: pt.System) -> Tuple[bool, bool]:
    """
    Test if copying a system creates independent copies.
    Returns (deepcopy_independent, native_copy_independent)
    """
    # Test copy.deepcopy
    system_deepcopy = copy.deepcopy(system)
    cue_ball_id = list(system.balls.keys())[0]
    original_pos = system.balls[cue_ball_id].state.rvw[0].copy()
    system_deepcopy.balls[cue_ball_id].state.rvw[0, 0] += 1.0
    deepcopy_independent = np.allclose(system.balls[cue_ball_id].state.rvw[0], original_pos)

    # Test native copy
    system_native = system.copy()
    system_native.balls[cue_ball_id].state.rvw[0, 1] += 1.0
    native_independent = np.allclose(system.balls[cue_ball_id].state.rvw[0], original_pos)

    return deepcopy_independent, native_independent


# ============================================================================
# Performance Benchmarks
# ============================================================================


def benchmark_copy_method(obj, copy_func: Callable, iterations: int = 100) -> float:
    """
    Benchmark a copy method by timing multiple iterations.
    Returns average time per copy in seconds.
    """
    start = time.perf_counter()
    for _ in range(iterations):
        copied = copy_func(obj)
        del copied  # Immediately delete to avoid memory accumulation
    end = time.perf_counter()
    return (end - start) / iterations


def run_performance_tests(iterations: int = 100):
    """Run performance benchmarks for all object types."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 80)

    # Test Ball - Simple
    print("\n--- Ball (Simple, no history) ---")
    ball_simple = create_simple_ball()
    time_deepcopy = benchmark_copy_method(ball_simple, copy.deepcopy, iterations)
    time_native = benchmark_copy_method(ball_simple, lambda b: b.copy(), iterations)
    print(f"copy.deepcopy:  {time_deepcopy * 1000:.4f} ms")
    print(f"native copy():  {time_native * 1000:.4f} ms")
    print(f"Speedup:        {time_deepcopy / time_native:.2f}x")
    del ball_simple

    # Test Ball - Complex
    print("\n--- Ball (Complex, history_length=100) ---")
    ball_complex = create_complex_ball(history_length=100)
    time_deepcopy = benchmark_copy_method(ball_complex, copy.deepcopy, iterations)
    time_native = benchmark_copy_method(ball_complex, lambda b: b.copy(), iterations)
    print(f"copy.deepcopy:  {time_deepcopy * 1000:.4f} ms")
    print(f"native copy():  {time_native * 1000:.4f} ms")
    print(f"Speedup:        {time_deepcopy / time_native:.2f}x")
    del ball_complex

    # Test Table
    print("\n--- Table (Standard) ---")
    table = create_simple_table()
    time_deepcopy = benchmark_copy_method(table, copy.deepcopy, iterations)
    time_native = benchmark_copy_method(table, lambda t: t.copy(), iterations)
    print(f"copy.deepcopy:  {time_deepcopy * 1000:.4f} ms")
    print(f"native copy():  {time_native * 1000:.4f} ms")
    print(f"Speedup:        {time_deepcopy / time_native:.2f}x")
    del table

    # Test System - Simple
    print("\n--- System (Simple, 2 balls, no history) ---")
    system_simple = create_simple_system(num_balls=2)
    time_deepcopy = benchmark_copy_method(system_simple, copy.deepcopy, iterations)
    time_native = benchmark_copy_method(system_simple, lambda s: s.copy(), iterations)
    print(f"copy.deepcopy:  {time_deepcopy * 1000:.4f} ms")
    print(f"native copy():  {time_native * 1000:.4f} ms")
    print(f"Speedup:        {time_deepcopy / time_native:.2f}x")
    del system_simple

    # Test System - Complex
    print("\n--- System (Complex, 16 balls, history_length=50) ---")
    system_complex = create_complex_system(num_balls=16, history_length=50)
    time_deepcopy = benchmark_copy_method(system_complex, copy.deepcopy, iterations=20)
    time_native = benchmark_copy_method(system_complex, lambda s: s.copy(), iterations=20)
    print(f"copy.deepcopy:  {time_deepcopy * 1000:.4f} ms")
    print(f"native copy():  {time_native * 1000:.4f} ms")
    print(f"Speedup:        {time_deepcopy / time_native:.2f}x")
    del system_complex


# ============================================================================
# Memory Profiling
# ============================================================================


def get_deep_size(obj, seen: Set[int] | None = None) -> int:
    """
    递归计算对象及其所有引用对象的总内存大小（字节）。

    参数:
        obj: 要测量的对象
        seen: 已访问对象的 id 集合（避免循环引用）

    返回:
        int: 总内存大小（字节）
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)

    # 避免重复计算和循环引用
    if obj_id in seen:
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    # 处理不同类型的对象
    if isinstance(obj, dict):
        # 字典：递归计算键和值
        size += sum(get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        # 序列和集合：递归计算元素
        size += sum(get_deep_size(item, seen) for item in obj)
    elif isinstance(obj, np.ndarray):
        # numpy 数组：使用 nbytes
        size += obj.nbytes
    elif hasattr(obj, "__dict__"):
        # 自定义对象：递归计算所有属性
        size += get_deep_size(obj.__dict__, seen)
    elif hasattr(obj, "__slots__"):
        # 使用 __slots__ 的对象
        size += sum(
            get_deep_size(getattr(obj, slot, None), seen)
            for slot in obj.__slots__
            if hasattr(obj, slot)
        )

    return size


def compare_object_memory(obj1: Any, obj2: Any, label: str = "Object") -> Tuple[int, int, float]:
    """
    比较两个对象的内存占用。

    返回: (obj1_size, obj2_size, ratio)
    """
    size1 = get_deep_size(obj1)
    size2 = get_deep_size(obj2)
    ratio = size1 / size2 if size2 > 0 else 1.0

    return size1, size2, ratio


def measure_copy_memory(obj, copy_func: Callable, iterations: int = 10) -> float:
    """
    Measure peak memory usage when copying an object.
    Returns peak memory in MB.
    """

    def copy_n_times():
        copies = []
        for _ in range(iterations):
            copies.append(copy_func(obj))
        return copies

    mem_usage = memory_usage(copy_n_times, max_usage=True)  # type: ignore
    return mem_usage


def run_memory_tests(iterations: int = 10):
    """Run memory profiling for all object types."""
    print("\n" + "=" * 80)
    print("MEMORY PROFILING - Object-Level Measurements")
    print("=" * 80)

    # Test Ball - Simple
    print("\n--- Ball (Simple, no history) ---")
    ball_simple = create_simple_ball()
    ball_deepcopy = copy.deepcopy(ball_simple)
    ball_native = ball_simple.copy()

    size_orig, _, _ = compare_object_memory(ball_simple, ball_simple, "Original")
    size_deepcopy, _, _ = compare_object_memory(ball_deepcopy, ball_deepcopy, "Deepcopy")
    size_native, _, _ = compare_object_memory(ball_native, ball_native, "Native")

    print(f"Original Ball:      {size_orig / 1024:.2f} KB")
    print(f"copy.deepcopy():    {size_deepcopy / 1024:.2f} KB")
    print(f"native copy():      {size_native / 1024:.2f} KB")
    print(f"Ratio (deep/native): {size_deepcopy / size_native:.2f}x")
    del ball_simple, ball_deepcopy, ball_native

    # Test Ball - Complex
    print("\n--- Ball (Complex, history_length=100) ---")
    ball_complex = create_complex_ball(history_length=100)
    ball_deepcopy = copy.deepcopy(ball_complex)
    ball_native = ball_complex.copy()

    size_orig = get_deep_size(ball_complex)
    size_deepcopy = get_deep_size(ball_deepcopy)
    size_native = get_deep_size(ball_native)

    print(f"Original Ball:      {size_orig / 1024:.2f} KB")
    print(f"copy.deepcopy():    {size_deepcopy / 1024:.2f} KB")
    print(f"native copy():      {size_native / 1024:.2f} KB")
    print(f"Ratio (deep/native): {size_deepcopy / size_native:.2f}x")

    # 检测共享对象
    print("\n共享对象检测:")
    print(f"  params 是否共享: {id(ball_complex.params) == id(ball_native.params)}")
    print(
        f"  initial_orientation 是否共享: {id(ball_complex.initial_orientation) == id(ball_native.initial_orientation)}"
    )
    print(f"  state.rvw 是否共享: {id(ball_complex.state.rvw) == id(ball_native.state.rvw)}")

    del ball_complex, ball_deepcopy, ball_native

    # Test Table
    print("\n--- Table (Standard) ---")
    table = create_simple_table()
    table_deepcopy = copy.deepcopy(table)
    table_native = table.copy()

    size_orig = get_deep_size(table)
    size_deepcopy = get_deep_size(table_deepcopy)
    size_native = get_deep_size(table_native)

    print(f"Original Table:     {size_orig / 1024:.2f} KB")
    print(f"copy.deepcopy():    {size_deepcopy / 1024:.2f} KB")
    print(f"native copy():      {size_native / 1024:.2f} KB")
    print(f"Ratio (deep/native): {size_deepcopy / size_native:.2f}x")
    del table, table_deepcopy, table_native

    # Test System - Simple
    print("\n--- System (Simple, 2 balls, no history) ---")
    system_simple = create_simple_system(num_balls=2)
    system_deepcopy = copy.deepcopy(system_simple)
    system_native = system_simple.copy()

    size_orig = get_deep_size(system_simple)
    size_deepcopy = get_deep_size(system_deepcopy)
    size_native = get_deep_size(system_native)

    print(f"Original System:    {size_orig / 1024:.2f} KB")
    print(f"copy.deepcopy():    {size_deepcopy / 1024:.2f} KB")
    print(f"native copy():      {size_native / 1024:.2f} KB")
    print(f"Ratio (deep/native): {size_deepcopy / size_native:.2f}x")
    del system_simple, system_deepcopy, system_native

    # Test System - Complex
    print("\n--- System (Complex, 16 balls, history_length=50) ---")
    system_complex = create_complex_system(num_balls=16, history_length=50)
    system_deepcopy = copy.deepcopy(system_complex)
    system_native = system_complex.copy()

    size_orig = get_deep_size(system_complex)
    size_deepcopy = get_deep_size(system_deepcopy)
    size_native = get_deep_size(system_native)

    print(f"Original System:    {size_orig / 1024:.2f} KB")
    print(f"copy.deepcopy():    {size_deepcopy / 1024:.2f} KB")
    print(f"native copy():      {size_native / 1024:.2f} KB")
    print(f"Ratio (deep/native): {size_deepcopy / size_native:.2f}x")

    # 检测球的 params 是否共享
    print("\n共享对象检测 (System):")
    cue_id = list(system_complex.balls.keys())[0]
    print(
        f"  球的 params 是否共享: {id(system_complex.balls[cue_id].params) == id(system_native.balls[cue_id].params)}"
    )
    print(f"  cue specs 是否共享: {id(system_complex.cue.specs) == id(system_native.cue.specs)}")

    del system_complex, system_deepcopy, system_native

    # Process-level memory comparison (原有的方法，作为对照)
    print("\n" + "=" * 80)
    print("MEMORY PROFILING - Process-Level Measurements (for reference)")
    print("=" * 80)

    # Test Ball - Complex
    print("\n--- Ball (Complex, history_length=100) ---")
    ball_complex = create_complex_ball(history_length=100)

    print("Measuring copy.deepcopy memory usage...")
    mem_deepcopy = measure_copy_memory(ball_complex, copy.deepcopy, iterations)

    print("Measuring native copy() memory usage...")
    mem_native = measure_copy_memory(ball_complex, lambda b: b.copy(), iterations)

    print(f"copy.deepcopy:  {mem_deepcopy:.2f} MB")
    print(f"native copy():  {mem_native:.2f} MB")
    print(f"Ratio:          {mem_deepcopy / mem_native:.2f}x")
    del ball_complex

    # Test System - Complex
    print("\n--- System (Complex, 16 balls, history_length=50) ---")
    system_complex = create_complex_system(num_balls=16, history_length=50)

    print("Measuring copy.deepcopy memory usage...")
    mem_deepcopy = measure_copy_memory(system_complex, copy.deepcopy, iterations=5)

    print("Measuring native copy() memory usage...")
    mem_native = measure_copy_memory(system_complex, lambda s: s.copy(), iterations=5)

    print(f"copy.deepcopy:  {mem_deepcopy:.2f} MB")
    print(f"native copy():  {mem_native:.2f} MB")
    print(f"Ratio:          {mem_deepcopy / mem_native:.2f}x")
    del system_complex


# ============================================================================
# Correctness Tests Runner
# ============================================================================


def run_correctness_tests():
    """Run correctness tests to verify copy independence."""
    print("\n" + "=" * 80)
    print("CORRECTNESS TESTS")
    print("=" * 80)

    # Test Ball
    print("\n--- Ball Copy Independence ---")
    ball = create_complex_ball(history_length=10)
    deepcopy_ok, native_ok = test_ball_copy_independence(ball)
    print(f"copy.deepcopy independent: {'✓ PASS' if deepcopy_ok else '✗ FAIL'}")
    print(f"native copy() independent: {'✓ PASS' if native_ok else '✗ FAIL'}")
    del ball

    # Test Table
    print("\n--- Table Copy Independence ---")
    table = create_simple_table()
    deepcopy_ok, native_ok = test_table_copy_independence(table)
    print(f"copy.deepcopy independent: {'✓ PASS' if deepcopy_ok else '✗ FAIL'}")
    print(f"native copy() independent: {'✓ PASS' if native_ok else '✗ FAIL'}")
    del table

    # Test System
    print("\n--- System Copy Independence ---")
    system = create_complex_system(num_balls=5, history_length=10)
    deepcopy_ok, native_ok = test_system_copy_independence(system)
    print(f"copy.deepcopy independent: {'✓ PASS' if deepcopy_ok else '✗ FAIL'}")
    print(f"native copy() independent: {'✓ PASS' if native_ok else '✗ FAIL'}")
    del system


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all comparison tests and generate report."""
    print("=" * 80)
    print("Pooltool Copy Method Comparison")
    print("copy.deepcopy vs native copy() methods")
    print("=" * 80)

    # Run all tests
    run_correctness_tests()
    run_performance_tests()
    run_memory_tests()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Based on the tests above:

1. CORRECTNESS: Both methods should create independent copies (verify above)

2. PERFORMANCE: Native copy() is expected to be faster due to:
   - Shared immutable objects (BallParams, BallOrientation, CueSpecs)
   - Optimized attrs.evolve implementation
   - Less allocation overhead

3. MEMORY: Native copy() is expected to use less memory due to:
   - Sharing frozen/immutable objects instead of duplicating them
   - Read-only numpy arrays may be shared
   - More efficient for large histories

RECOMMENDATION:
If correctness tests pass and performance/memory show significant improvements,
consider migrating the codebase from copy.deepcopy to native copy() methods in:
- src/eval/poolenv.py
- src/train/poolenv.py  
- src/train/agents/geometry.py
    """)


if __name__ == "__main__":
    main()
