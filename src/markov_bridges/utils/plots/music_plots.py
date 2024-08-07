from matplotlib import pyplot as plt



def plot_songs(original_sample,
               generative_sample,
               song_1_index = 0,
               song_2_index = 1,
               number_of_steps = 64,
               repeat_sample = 3,
               conditional_dimension = 32,
               save_path="songs.png",
               repeating=True,
    ):
    if repeating:
        get_song_index = lambda song_index,repeat_sample,example_index:song_index*repeat_sample+example_index
    prepare_song = lambda x,number_of_steps:x[:number_of_steps].cpu().numpy()
    number_of_steps = 2*conditional_dimension
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 5)) # Create a 2x3 grid of subplots

    if repeating:
        example_1_index_1 = get_song_index(song_1_index,repeat_sample,0)
        example_1_index_2 = get_song_index(song_1_index,repeat_sample,1)
    else:
        example_1_index_1 = song_1_index
        example_1_index_2 = song_2_index

    axs[0,0].set_title("Real")
    axs[0,0].plot(prepare_song(original_sample[example_1_index_1],number_of_steps),"^",label="original")
    axs[0,0].grid("True")
    axs[0,0].axvline(x=conditional_dimension, color='r', linestyle='--', label='condition')  # Add vertical line
    axs[0,0].set_ylabel("Song 2")
    axs[0,0].set_xticklabels([])  # Remove x-axis tick labels
    axs[0,0].set_yticklabels([])  # Remove y-axis tick labels

    axs[0,1].set_title("Example 1")
    axs[0,1].plot(prepare_song(generative_sample[example_1_index_1],number_of_steps),"*",label="generative")
    axs[0,1].grid("True")
    axs[0,1].axvline(x=conditional_dimension, color='r', linestyle='--', label='condition')  # Add vertical line
    axs[0,1].set_xticklabels([])  # Remove x-axis tick labels
    axs[0,1].set_yticklabels([])  # Remove y-axis tick labels

    axs[0,2].set_title("Example 2")
    axs[0,2].plot(prepare_song(generative_sample[example_1_index_2],number_of_steps),"*",label="generative")
    axs[0,2].grid("True")
    axs[0,2].axvline(x=conditional_dimension, color='r', linestyle='--', label='condition')  # Add vertical line
    axs[0,2].set_xticklabels([])  # Remove x-axis tick labels
    axs[0,2].set_yticklabels([])  # Remove y-axis tick labels

    if repeating:
        example_2_index_1 = get_song_index(song_2_index,repeat_sample,0)
        example_2_index_2 = get_song_index(song_2_index,repeat_sample,1)
    else:
        example_2_index_1 = song_1_index
        example_2_index_2 = song_2_index

    axs[1,0].plot(prepare_song(original_sample[example_2_index_1],number_of_steps),"^",label="original")
    axs[1,0].grid("True")
    axs[1,0].axvline(x=conditional_dimension, color='r', linestyle='--', label='condition')  # Add vertical line
    axs[1,0].set_ylabel("Song 2")
    axs[1,0].set_xticklabels([])  # Remove x-axis tick labels
    axs[1,0].set_yticklabels([])  # Remove y-axis tick labels
    axs[1,0].set_xlabel("Bar")

    axs[1,1].plot(prepare_song(generative_sample[example_2_index_1],number_of_steps),"*",label="generative")
    axs[1,1].grid("True")
    axs[1,1].axvline(x=conditional_dimension, color='r', linestyle='--', label='condition')  # Add vertical line
    axs[1,1].set_xticklabels([])  # Remove x-axis tick labels
    axs[1,1].set_yticklabels([])  # Remove y-axis tick labels
    axs[1,1].set_xlabel("Bar")

    axs[1,2].plot(prepare_song(generative_sample[example_2_index_2],number_of_steps),"*",label="generative")
    axs[1,2].axvline(x=conditional_dimension, color='r', linestyle='--', label='condition')  # Add vertical line
    axs[1,2].set_xticklabels([])  # Remove x-axis tick labels
    axs[1,2].set_yticklabels([])  # Remove y-axis tick labels
    axs[1,2].grid("True")
    axs[1,2].set_xlabel("Bar")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()