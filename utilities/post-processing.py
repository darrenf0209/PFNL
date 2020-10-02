import os
import json
import numpy as np
import glob
import math
import matplotlib.pyplot as plt




# Append each JSON object from log file reading line by line
def append_logs(log_file):
    logs = []
    with open(log_file) as lf:
        for line in lf:
            logs.append(json.loads(line))
    return logs


def retrieve_model_names(path):
    files = os.listdir(path)
    # print(log_files)
    names = []
    for file in range(len(files)):
        name = files[file].rstrip('.txt')
        # name = name.lstrip('/content/pfnl_')
        names.append(name)
    # print(names)
    return sorted(names)


# This function returns a list of duplicates of an array
def list_duplicates(arr):
    duplicates = set([x for x in arr if arr.count(x) > 1])
    if list(duplicates):
        print("{} duplicates found".format(len(duplicates)))
        return sorted(duplicates)
    else:
        print("No duplicates found after cleaning")
        return 0


def clean_logs(logs):
    iterations = []
    train_times = []
    PSNRs = []
    losses = []
    count_nan = 0
    for log in logs:
        # Skip the log where Loss is NaN.
        # Incorrect reading due to initial loading of model or crash after checkpoint
        if math.isnan(log['Loss']):
            # print("NaN detected at iteration: {}".format(log['Iteration']))
            count_nan += 1
            continue
        else:
            iterations.append(log['Iteration'])
            train_times.append(log['Training Time'])
            PSNRs.append(log['PSNR'])
            losses.append(log['Loss'])
    print("{} NaNs removed during cleaning".format(count_nan))
    # duplicates = list_duplicates(iterations)
    # if not duplicates:
    dict_out = {
        "Iterations": iterations,
        "Train Times": train_times,
        "PSNRs": PSNRs,
        "Losses": losses
    }
    return dict_out


def hours_mins_secs(seconds):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return "%d hours, %02d minutes and %02d seconds" % (hours, mins, secs)


def convert_to_numpy(data):
    key_names = []
    for key in data:
        data[key] = np.array(data[key])
        key_names.append(key)
    # print("Key names identified: {}".format(key_names))
    return data


def timing_summary(np_data):
    # Timing characteristics
    time_to_finish = np.cumsum(np_data['Train Times'])[-1]
    avg_train_time = np.mean(np_data['Train Times'])
    return time_to_finish, avg_train_time


def best_model(np_data):
    # Find the best performing model
    max_ind = np.argmax(np_data['PSNRs'])
    time_to_max = np.cumsum(np_data['Train Times'][0:max_ind])[-1]
    iteration_max = np_data['Iterations'][max_ind]
    PSNR_max = np_data['PSNRs'][max_ind]
    return iteration_max, PSNR_max, time_to_max

def lowest_loss(np_data):
    # Find the best performing model
    min_ind = np.argmin(np_data['Losses'])
    iteration_min = np_data['Losses'][min_ind]
    loss_min = np_data['Losses'][min_ind]
    return iteration_min, loss_min


def top_N(np_data, N=3):
    # Retrieve the top N best performing models
    top_indices = np.argpartition(np_data['PSNRs'], -N)[-N:]
    # print(top_indices)
    top_N_iterations = np_data['Iterations'][top_indices]
    # print(top_N_iterations)
    top_N_PSNRs = np_data['PSNRs'][top_indices]
    # print(top_N_PSNRs)
    return top_N_iterations, top_N_PSNRs

def bot_N(np_data, N=3):
    # Retrieve the top N best performing models
    bot_indices = np.argpartition(-np_data['Losses'], -N)[-N:]
    # print(top_indices)
    bot_N_iterations = np_data['Iterations'][bot_indices]
    # print(top_N_iterations)
    bot_N_losses = np_data['Losses'][bot_indices]
    # print(top_N_PSNRs)
    return bot_N_iterations, bot_N_losses


''' This function returns the n-point moving average
'''


def N_moving_avg(np_data, N=5):
    # calculate average based on first N point
    PSNRs = np_data['PSNRs']
    window_size = N

    i = 0
    moving_avgs = []

    while i < len(PSNRs) - window_size + 1:
        cur_window = PSNRs[i: i + window_size]
        cur_window_avg = sum(cur_window) / window_size

        moving_avgs.append(cur_window_avg)

        i += 1

    plt.plot(moving_avgs, label='{}-point moving average'.format(N))
    plt.legend(loc='lower right')
    # print(moving_avgs)


def plot_iter_vs_PSNR(np_data, iteration_max, PSNR_max, name='Model'):
    scaled_iterations = np_data['Iterations'] / 1000
    PSNRs = np_data['PSNRs']
    scaled_iteration_max = iteration_max / 1000
    plt.plot(scaled_iterations, PSNRs)
    # plt.plot(scaled_iteration_max, PSNR_max, 'o',
    #          markersize=10, fillstyle='none',
    #          label='Max ({}, {:.2f})'.format(scaled_iteration_max, PSNR_max))
    plt.plot(scaled_iteration_max, PSNR_max, 'o', markersize=10, fillstyle='none')
    plt.legend(loc='lower right')
    plt.xlabel("Iterations ('000s)")
    plt.ylabel('PSNR (dB)')
    plt.title('{}: PSNR vs Iterations'.format(name))
    plt.axis([0, 250, 25, 35])
    plt.savefig(os.path.join(save_path, '{} PSNR.png').format(name))
    plt.close()
    # plt.show()

def plot_iter_vs_loss(np_data, iteration_min, loss_min, name='Model'):
    scaled_iterations = np_data['Iterations'] / 1000
    losses = np_data['Losses']
    scaled_iteration_min = iteration_min / 1000
    plt.plot(scaled_iterations, losses, 'g')
    # plt.plot(scaled_iteration_max, PSNR_max, 'o',
    #          markersize=10, fillstyle='none',
    #          label='Max ({}, {:.2f})'.format(scaled_iteration_max, PSNR_max))
    plt.plot(scaled_iteration_min, loss_min, 'co', markersize=10, fillstyle='none')
    plt.legend(loc='lower right')
    plt.xlabel("Iterations ('000s)")
    plt.ylabel('Loss')
    plt.title('{}: Loss vs Iterations'.format(name))
    # plt.axis([0, 250, 0, 5])
    plt.savefig(os.path.join(save_path, '{} loss.png').format(name))
    plt.close()
    # plt.show()

def gen_stats(np_data, name='Model', graph=True):
    print("Summary statistics for {}".format(name))
    # np_data = convert_to_numpy(data)
    time_to_finish, avg_train_time = timing_summary(np_data)
    print("Total training time: {}".format(hours_mins_secs(time_to_finish)))
    print("Average training time per checkpoint: {:.2f}s".format(avg_train_time))
    iteration_max, PSNR_max, time_to_max = best_model(np_data)
    iteration_min, loss_min = lowest_loss(np_data)
    print("Best performing model from {}".format(name))
    print("Iterations: {}, PSNR: {:.2f} dB, Cumulative Training Time: {}"
          .format(iteration_max, PSNR_max, hours_mins_secs(time_to_max)))
    top_N_iterations, top_N_PSNRs = top_N(np_data, N=3)
    bot_N_iterations, bot_N_losses = bot_N(np_data, N=3)
    if graph:
        plot_iter_vs_PSNR(np_data, top_N_iterations, top_N_PSNRs, name)
        plot_iter_vs_loss(np_data, bot_N_iterations, bot_N_losses, name)
        # plot_iter_vs_PSNR(np_data, iteration_max, PSNR_max, name)


def multiplot(y, names, colors):
    for i in range(len(y)):
        plt.plot(y[i], colors[i], label=names[i])
    plt.xlabel("Iterations ('000s)")
    plt.ylabel('PSNR (dB)')
    plt.title('Mean of models: PSNR vs Iterations'.format(names))
    plt.legend(loc='lower right')
    plt.axis([0, 250, 25, 35])
    plt.show()


def multiscatter(x, y, names, colors):
    for i in range(len(x)):
        print(names[i], x[i], y[i])
        plt.scatter(x[i], y[i], s=50, c=colors[i],
                    label="{} ({:.2f}, {:.2f})".format(names[i], x[i], y[i]))
    plt.xlabel("Total Training Time (hours)")
    plt.ylabel('Max PSNR (dB)')
    plt.title('Total Training Time vs Max PSNR'.format(names))
    plt.legend(loc='lower right')
    plt.show()

save_path = 'report/obj2/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
logs_path = 'logs/obj2/'
model_names = retrieve_model_names(logs_path)
print(model_names)
log_files = sorted(glob.glob(os.path.join(logs_path, '*.txt')))

dict_logs = {}
for item in range(len(log_files)):
    # print(model_names[item])
    dict_logs[model_names[item]] = append_logs(log_files[item])
    # print(model_names[item])

#
for i in sorted(dict_logs.keys()):
    # print("Before: {}".format(dict_logs[i]))
    dict_logs[i] = clean_logs(dict_logs[i])
    dict_logs[i] = convert_to_numpy(dict_logs[i])
    # N_moving_avg(dict_logs[i], N=10)
    # print("After: {}".format(dict_logs[i]))
    # top_N(dict_logs[i])
    gen_stats(dict_logs[i], name=i, graph=True)
#
# """Multiple figures per graph"""

arr_of_iters = []
arr_of_PSNRs = []
cumulative_times = []
max_PSNRs = []
for i in sorted(dict_logs.keys()):
    # print(i)
    # print(dict_logs[i]['Iterations'])
    arr_of_iters.append(dict_logs[i]['Iterations'] / 1000)
    arr_of_PSNRs.append(dict_logs[i]['PSNRs'])
    finish_time = timing_summary(dict_logs[i])
    finish_time = finish_time[0] / 3600  # hours
    cumulative_times.append(finish_time)
    # print(finish_time)
    PSNR_max = best_model(dict_logs[i])
    # print(PSNR_max[1])
    max_PSNRs.append(PSNR_max[1])

# print(cumulative_times)
# orange, green
colors = ['#FF8300', '#CD7A21', '#B2712C', '#32FF00', '#45C425', '#469332', '#00FFE8', '#2ACBBC', '#429890', '#0093FF', '#3485C0', '#417092', '#C100FF', '#9636B5', '#885E95']
# multiplot(arr_of_iters, arr_of_PSNRs, model_names, colors)
# multiscatter(cumulative_times, max_PSNRs, model_names, colors)

# alt_sum_mean = (arr_of_PSNRs[0] + arr_of_PSNRs[1][1:500] + arr_of_PSNRs[2][1:500]) / 3
# control_3_mean = (arr_of_PSNRs[3][1:500] + arr_of_PSNRs[4][1:500]  + arr_of_PSNRs[5][1:500] ) / 3
# control_5_mean = (arr_of_PSNRs[6][1:500]  + arr_of_PSNRs[7][1:500]  + arr_of_PSNRs[8][1:500] ) / 3
# control_7_mean = (arr_of_PSNRs[9][1:500]  + arr_of_PSNRs[10][1:500] + arr_of_PSNRs[11][1:500] ) / 3
# null_mean = (arr_of_PSNRs[12][1:500]  + arr_of_PSNRs[13][1:500]  + arr_of_PSNRs[14][1:500] ) / 3
# multiplot_arr = [alt_sum_mean, control_3_mean, control_5_mean, control_7_mean, null_mean]
# colors = ['#FF8300', '#32FF00', '#00FFE8', '#0093FF', '#C100FF']
# model_names = ['Alternative', 'Control 3', 'Control 5', 'Control 7', 'Null']
# multiplot(multiplot_arr, model_names, colors)