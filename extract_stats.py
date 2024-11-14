import re
import statistics

def parse_log_file(filename):
    """Parses a log file and extracts CPU and GPU times.

    Args:
        filename (str): The name of the log file.

    Returns:
        list: A list of tuples, where each tuple contains the run number, CPU time, and GPU time.
    """

    results = []
    with open(filename, 'r') as f:
        run_num = 1
        for line in f:
            if "CPU time:" in line:
                cpu_time = float(re.search(r"CPU time: (\d+\.\d+)", line).group(1))
            elif "GPU Time:" in line:
                gpu_time = float(re.search(r"GPU Time: (\d+\.\d+)", line).group(1))
                results.append((run_num, cpu_time, gpu_time))
                run_num += 1

    return results

def calculate_stats(results):
    """Calculates the mean and standard deviation of CPU and GPU times, excluding the fastest and slowest times.

    Args:
        results (list): A list of tuples, as returned by `parse_log_file`.

    Returns:
        tuple: A tuple containing the mean and standard deviation of CPU times, and the mean and standard deviation of GPU times.
    """

    cpu_times = [cpu_time for _, cpu_time, _ in results]
    gpu_times = [gpu_time for _, _, gpu_time in results]

    # Remove the fastest and slowest times
    cpu_times.remove(max(cpu_times))
    cpu_times.remove(min(cpu_times))
    gpu_times.remove(max(gpu_times))
    gpu_times.remove(min(gpu_times))

    mean_cpu = statistics.mean(cpu_times)
    std_dev_cpu = statistics.stdev(cpu_times)
    mean_gpu = statistics.mean(gpu_times)
    std_dev_gpu = statistics.stdev(gpu_times)

    return mean_cpu, std_dev_cpu, mean_gpu, std_dev_gpu

if __name__ == "__main__":
    log_file = "experiment_12.log"  # Replace with your actual log file name
    results = parse_log_file(log_file)
    mean_cpu, std_dev_cpu, mean_gpu, std_dev_gpu = calculate_stats(results)

    print(f"Mean CPU time: {mean_cpu:.3f} msec, Mean GPU time: {mean_gpu:.3f} msec")
    print(f"Std Dev: {std_dev_cpu:.3f}, Std Dev: {std_dev_gpu:.3f}")