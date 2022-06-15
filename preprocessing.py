# Write features into required format by this repo

import os
import numpy as np




# seed = 0
# epoch = 147

# seed = 1
# epoch = 181

# seed = 2
# epoch = 195

# seed = 3
# epoch = 142

seed = 4
epoch = 120

model_base_dir = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/" \
                 "msd_net/2022-02-13/known_only_cross_entropy"

# Paths for all the features
reduced_train_feature_path = model_base_dir + "/seed_" + str(seed) + \
                             "/test_results/train_known_known_epoch_" + \
                             str(epoch) + "_features_reduced.npy"
reduced_test_known_feature_p0_path = model_base_dir + "/seed_" + str(seed) + \
                                  "/test_results/test_known_known_epoch_" + \
                                  str(epoch) + "_features_p0_reduced.npy"
reduced_test_known_feature_p1_path = model_base_dir + "/seed_" + str(seed) + \
                                  "/test_results/test_known_known_epoch_" + \
                                  str(epoch) + "_features_p1_reduced.npy"
reduced_test_known_feature_p2_path = model_base_dir + "/seed_" + str(seed) + \
                                  "/test_results/test_known_known_epoch_" + \
                                  str(epoch) + "_features_p2_reduced.npy"
reduced_test_known_feature_p3_path = model_base_dir + "/seed_" + str(seed) + \
                                  "/test_results/test_known_known_epoch_" + \
                                  str(epoch) + "_features_p3_reduced.npy"
reduced_test_unknown_feature_path = model_base_dir + "/seed_" + str(seed) + \
                                    "/test_results/test_unknown_unknown_epoch_" + \
                                    str(epoch) + "_features_reduced.npy"

# Path for all labels
train_label_path =  model_base_dir + "/seed_" + str(seed) + "/train_known_known_epoch_" + \
                    str(epoch) + "_labels.npy"
test_known_p0_label_path = model_base_dir + "/seed_" + str(seed) + \
                            "/test_results/test_known_known_epoch_" + \
                            str(epoch) + "_part_0_labels.npy"
test_known_p1_label_path = model_base_dir + "/seed_" + str(seed) + \
                            "/test_results/test_known_known_epoch_" + \
                            str(epoch) + "_part_1_labels.npy"
test_known_p2_label_path = model_base_dir + "/seed_" + str(seed) + \
                            "/test_results/test_known_known_epoch_" + \
                            str(epoch) + "_part_2_labels.npy"
test_known_p3_label_path = model_base_dir + "/seed_" + str(seed) + \
                            "/test_results/test_known_known_epoch_" + \
                            str(epoch) + "_part_3_labels.npy"

# Load all features
reduced_train_feature = np.load(reduced_train_feature_path)
reduced_test_known_feature_p0 = np.load(reduced_test_known_feature_p0_path)
reduced_test_known_feature_p1 = np.load(reduced_test_known_feature_p1_path)
reduced_test_known_feature_p2 = np.load(reduced_test_known_feature_p2_path)
reduced_test_known_feature_p3 = np.load(reduced_test_known_feature_p3_path)
reduced_test_unknown_feature = np.load(reduced_test_unknown_feature_path)

reduced_test_known_feature = np.concatenate((reduced_test_known_feature_p0,
                                             reduced_test_known_feature_p1,
                                             reduced_test_known_feature_p2,
                                             reduced_test_known_feature_p3), axis=0)

# Load all labels
train_labels = np.load(train_label_path)
test_known_p0_label = np.load(test_known_p0_label_path)
test_known_p1_label = np.load(test_known_p1_label_path)
test_known_p2_label = np.load(test_known_p2_label_path)
test_known_p3_label = np.load(test_known_p3_label_path)

test_known_labels = np.concatenate((test_known_p0_label,
                                   test_known_p1_label,
                                   test_known_p2_label,
                                   test_known_p3_label), axis=0)

# Check shapes
print("train feature: ", reduced_train_feature.shape)
print("train label: ", train_labels.shape)
print("test known feature: ", reduced_test_known_feature.shape)
print("test known labels: ", test_known_labels.shape)
print("test unknown feature: ", reduced_test_unknown_feature.shape)

# Save reformatted feature
feature_save_base = model_base_dir + "/seed_" + str(seed)




def reformat_feature(original_known_feature,
                     original_label,
                     feature_save_path,
                     original_unknown_feature=None,
                     nb_known_class=293):
    """

    :param original_known_feature:
    :param original_label:
    :param feature_save_path:
    :param original_unknown_feature:
    :param nb_known_class:
    :return:
    """
    formatted_feature = ""

    with open(feature_save_path, 'w') as f:
        for i in range(original_label.shape[0]):
            one_label = original_label[i]
            one_line = str(one_label) + " "

            for j in range(original_known_feature.shape[1]):
                index = j
                one_feature = original_known_feature[i][j]
                one_entry_pair = str(index) + ":" + str(one_feature) + " "

                one_line += one_entry_pair

            formatted_feature += one_line + "\n"

        if original_unknown_feature is not None:
            for i in range(original_unknown_feature.shape[0]):
                one_label = nb_known_class
                one_line = str(one_label) + " "

                for j in range(original_unknown_feature.shape[1]):
                    index = j
                    one_feature = original_unknown_feature[i][j]
                    one_entry_pair = str(index) + ":" + str(one_feature) + " "

                    one_line += one_entry_pair

                formatted_feature += one_line + "\n"

        f.write(formatted_feature)

    print("File saved to: ", feature_save_path)




def gen_debug_feature(original_known_feature,
                      original_label,
                      feature_save_path,
                      nb_known_class=293):
    """

    :param original_known_feature:
    :param original_label:
    :param feature_save_path:
    :return:
    """
    formatted_feature = ""
    label_list = {}

    for i in range(nb_known_class):
        label_list[i] = 0

    print(label_list)

    with open(feature_save_path, 'w') as f:
        for i in range(original_label.shape[0]):
            one_label = original_label[i]
            one_line = str(one_label) + " "

            if label_list[one_label] != 3:
                label_list[one_label] += 1

                for j in range(original_known_feature.shape[1]):
                    index = j
                    one_feature = original_known_feature[i][j]
                    one_entry_pair = str(index) + ":" + str(one_feature) + " "

                    one_line += one_entry_pair

                formatted_feature += one_line + "\n"

            else:
                pass

        f.write(formatted_feature)

    print("File saved to: ", feature_save_path)




if __name__ == "__main__":
    reformat_feature(original_known_feature=reduced_train_feature,
                     original_label=train_labels,
                     feature_save_path=os.path.join(feature_save_base, "train_feature"))

    reformat_feature(original_known_feature=reduced_test_known_feature,
                     original_label=test_known_labels,
                     original_unknown_feature=reduced_test_unknown_feature,
                     feature_save_path=os.path.join(feature_save_base, "test_feature"))

    # gen_debug_feature(original_known_feature=reduced_train_feature,
    #                   original_label=train_labels,
    #                   feature_save_path=os.path.join(feature_save_base, "train_feature_debug"))