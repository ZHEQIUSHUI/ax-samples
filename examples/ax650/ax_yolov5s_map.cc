/*
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

/*
 * Author: hebing
 */

#include <cstdio>
#include <cstring>
#include <numeric>

#include <opencv2/opencv.hpp>
#include "base/common.hpp"
#include "base/detection.hpp"
#include "middleware/io.hpp"

#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include <sys/fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <sys/stat.h>
#include <iostream>

const char* CLASS_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

const int coco80_to_coco91[90] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

const float ANCHORS[18] = {
    10, 13, 16, 30, 33, 23,
    30, 61, 62, 45, 59, 119,
    116, 90, 156, 198, 373, 326};

const int DEFAULT_IMG_H = 640;
const int DEFAULT_IMG_W = 640;

const float PROB_THRESHOLD = 0.001; // 0.001f; //0.25f;
const float NMS_THRESHOLD = 0.6f;

void create_directory(const std::string& directory)
{
    size_t pos = 0;
    std::string dir = directory;
    if (dir[dir.size() - 1] != '/')
    {
        dir += '/';
    }
    while ((pos = dir.find_first_of('/', pos + 1)) != std::string::npos)
    {
        mkdir(dir.substr(0, pos).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

int scan_one_dir(const char* dir_name, std::vector<std::string>& files_vector)
{
    files_vector.clear();

    if (NULL == dir_name)
    {
        printf("dir_name is null !!!\n");
        return -1;
    }

    struct stat s;
    lstat(dir_name, &s);
    if (!S_ISDIR(s.st_mode))
    {
        printf("dir_name is not dir !!!\n");
        return -1;
    }

    struct dirent* filename = NULL;
    DIR* dir = NULL;
    dir = opendir(dir_name);
    if (NULL == dir)
    {
        printf("opendir failed !!!\n");
        return -1;
    }

    while ((filename = readdir(dir)) != NULL)
    {
        if (strcmp(filename->d_name, ".") == 0 || strcmp(filename->d_name, "..") == 0)
            continue;

        char wholePath[128] = {0};
        sprintf(wholePath, "%s", filename->d_name);
        files_vector.push_back(wholePath);
    }

    return 0;
}

namespace detection
{
    static inline float round_bit(double x, int bit)
    {
        int s = int(pow(10, bit));
        return float(round(x * s) / s);
    }

    void get_out_bbox_map(std::vector<Object>& proposals, std::vector<Object>& objects, const float nms_threshold, int letterbox_rows, int letterbox_cols, int src_rows, int src_cols, int class_num = 80, int max_det = 300)
    {
        qsort_descent_inplace(proposals);
        std::vector<int> picked;

        std::vector<std::vector<Object> > sel_proposals(class_num);
        std::vector<std::vector<int> > sel_picked(class_num);
        for (int i = 0; i < class_num; i++)
        {
            for (const auto val : proposals)
            {
                if (val.label == i)
                {
                    sel_proposals[i].push_back(val);
                }
            }
        }

        for (int i = 0; i < class_num; i++)
        {
            nms_sorted_bboxes(sel_proposals[i], sel_picked[i], nms_threshold);
        }

        std::vector<Object> new_proposals;
        for (int i = 0; i < class_num; i++)
        {
            for (int j = 0; j < sel_picked[i].size(); j++)
            {
                new_proposals.push_back(sel_proposals[i][sel_picked[i][j]]);
            }
        }

        picked.resize(new_proposals.size());
        for (int i = 0; i < new_proposals.size(); i++)
        {
            picked[i] = i;
        }

        /* yolov5 draw the result */
        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / src_rows) < (letterbox_cols * 1.0 / src_cols))
        {
            scale_letterbox = letterbox_rows * 1.0 / src_rows;
        }
        else
        {
            scale_letterbox = letterbox_cols * 1.0 / src_cols;
        }
        resize_cols = int(scale_letterbox * src_cols);
        resize_rows = int(scale_letterbox * src_rows);

        int tmp_h = (letterbox_rows - resize_rows) / 2;
        int tmp_w = (letterbox_cols - resize_cols) / 2;

        float ratio_x = (float)src_rows / resize_rows;
        float ratio_y = (float)src_cols / resize_cols;

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = new_proposals[picked[i]];
            float x0 = (objects[i].rect.x);
            float y0 = (objects[i].rect.y);
            float x1 = (objects[i].rect.x + objects[i].rect.width);
            float y1 = (objects[i].rect.y + objects[i].rect.height);

            x0 = (x0 - tmp_w) * ratio_x;
            y0 = (y0 - tmp_h) * ratio_y;
            x1 = (x1 - tmp_w) * ratio_x;
            y1 = (y1 - tmp_h) * ratio_y;

            x0 = std::max(std::min(x0, (float)(src_cols - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(src_rows - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(src_cols - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(src_rows - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
    }

    static void generate_proposals_yolov5s_map(int stride, const float* feat, float prob_threshold, std::vector<Object>& objects,
                                               int letterbox_cols, int letterbox_rows, const float* anchors, float prob_threshold_unsigmoid, bool multy_label = false)
    {
        int anchor_num = 3;
        int feat_w = letterbox_cols / stride;
        int feat_h = letterbox_rows / stride;
        int cls_num = 80;
        int anchor_group;
        if (stride == 8)
            anchor_group = 1;
        if (stride == 16)
            anchor_group = 2;
        if (stride == 32)
            anchor_group = 3;

        auto feature_ptr = feat;

        for (int h = 0; h <= feat_h - 1; h++)
        {
            for (int w = 0; w <= feat_w - 1; w++)
            {
                for (int a = 0; a <= anchor_num - 1; a++)
                {
                    double box_score = feature_ptr[4];
                    float dx = sigmoid(feature_ptr[0]);
                    float dy = sigmoid(feature_ptr[1]);
                    float dw = sigmoid(feature_ptr[2]);
                    float dh = sigmoid(feature_ptr[3]);
                    float pred_cx = (dx * 2.0f - 0.5f + w) * stride;
                    float pred_cy = (dy * 2.0f - 0.5f + h) * stride;
                    float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                    float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
                    float pred_w = dw * dw * 4.0f * anchor_w;
                    float pred_h = dh * dh * 4.0f * anchor_h;
                    float x0 = pred_cx - pred_w * 0.5f;
                    float y0 = pred_cy - pred_h * 0.5f;
                    float x1 = pred_cx + pred_w * 0.5f;
                    float y1 = pred_cy + pred_h * 0.5f;

                    if (multy_label)
                    {
                        for (int s = 0; s <= cls_num - 1; s++)
                        {
                            double score = feature_ptr[s + 5];

                            double final_score = sigmoid(box_score) * sigmoid(score);

                            if (final_score >= prob_threshold)
                            {
                                Object obj;
                                obj.rect.x = x0;
                                obj.rect.y = y0;
                                obj.rect.width = x1 - x0;
                                obj.rect.height = y1 - y0;
                                obj.label = s;
                                obj.prob = final_score;
                                objects.push_back(obj);
                            }
                        }
                    }
                    else
                    {
                        // process cls score
                        int class_index = 0;
                        double class_score = -FLT_MAX;
                        for (int s = 0; s <= cls_num - 1; s++)
                        {
                            double score = feature_ptr[s + 5];
                            if (score > class_score)
                            {
                                class_index = s;
                                class_score = score;
                            }
                        }
                        // process box score

                        double final_score = sigmoid(box_score) * sigmoid(class_score);

                        if (final_score > prob_threshold)
                        {
                            Object obj;
                            obj.rect.x = x0;
                            obj.rect.y = y0;
                            obj.rect.width = x1 - x0;
                            obj.rect.height = y1 - y0;
                            obj.label = class_index;
                            obj.prob = final_score;
                            objects.push_back(obj);
                        }
                    }

                    feature_ptr += (cls_num + 5);
                }
            }
        }
    }
} // namespace detection

void post_process(AX_ENGINE_IO_INFO_T* io_info, AX_ENGINE_IO_T* io_data,
                  std::array<int, 2> input_size, std::vector<detection::Object>& objects, int src_h, int src_w)
{
    std::vector<detection::Object> proposals;

    float prob_threshold_u_sigmoid = -1.0f * (float)std::log((1.0f / PROB_THRESHOLD) - 1.0f);
    timer timer_postprocess;
    for (uint32_t i = 0; i < io_info->nOutputSize; ++i)
    {
        auto& output = io_data->pOutputs[i];
        auto& info = io_info->pOutputs[i];
        auto ptr = (float*)output.pVirAddr;
        int32_t stride = (1 << i) * 8;
        detection::generate_proposals_yolov5s_map(stride, ptr, PROB_THRESHOLD, proposals, input_size[1],
                                                  input_size[0], ANCHORS, prob_threshold_u_sigmoid, true);
    }

    detection::get_out_bbox_map(proposals, objects, NMS_THRESHOLD, input_size[0], input_size[1],
                                src_h, src_w);

    return;
}

bool run_model(const std::string& model, std::string images_dir, std::array<int, 2> input_size, std::string save_dir)
{
    // 1. init engine
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    auto ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret)
    {
        return ret;
    }

    // 2. load model
    std::vector<char> model_buffer;
    if (!utilities::read_file(model, model_buffer))
    {
        fprintf(stderr, "Read Run-Joint model(%s) file failed.\n", model.c_str());
        return false;
    }

    // 3. create handle
    AX_ENGINE_HANDLE handle;
    ret = AX_ENGINE_CreateHandle(&handle, model_buffer.data(), model_buffer.size());
    SAMPLE_AX_ENGINE_DEAL_HANDLE
    fprintf(stdout, "Engine creating handle is done.\n");

    // 4. create context
    ret = AX_ENGINE_CreateContext(handle);
    SAMPLE_AX_ENGINE_DEAL_HANDLE
    fprintf(stdout, "Engine creating context is done.\n");

    // 5. set io
    AX_ENGINE_IO_INFO_T* io_info;
    ret = AX_ENGINE_GetIOInfo(handle, &io_info);
    SAMPLE_AX_ENGINE_DEAL_HANDLE
    fprintf(stdout, "Engine get io info is done. \n");

    // 6. alloc io
    AX_ENGINE_IO_T io_data;
    ret = middleware::prepare_io(io_info, &io_data, std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED));
    SAMPLE_AX_ENGINE_DEAL_HANDLE
    fprintf(stdout, "Engine alloc io is done. \n");

    std::vector<std::string> files_vector;
    if (0 != scan_one_dir(images_dir.c_str(), files_vector))
    {
        printf("scan image failed !!!");
        return false;
    }

    int count = 0;
    double total_time = 0.0;
    int warmup_count = 50;

    //    nlohmann::json predict_results;
    std::string coco_json_file = save_dir + "/" + "ax650_coco_results.json";
    printf("coco results: %s \n", coco_json_file.c_str());
    std::vector<uint8_t> input_image(input_size[0] * input_size[1] * 3, 0);
    //    std::ofstream json_file(coco_json_file);

    FILE* file_handle = fopen(coco_json_file.c_str(), "w");
    fprintf(file_handle, "[");
    bool is_first = true;

    for (auto file_name : files_vector)
    {
        printf("count: %d\n", count);

        std::string image_path = images_dir + "/" + file_name;
        std::string file_name_index = file_name.substr(0, file_name.rfind("."));
        printf("image path: %s image index: %s\n", image_path.c_str(), file_name_index.c_str());
        // if use jpg as input:
        cv::Mat src_image = cv::imread(image_path, 1);
        if (!src_image.data)
        {
            std::cout << "read image " << image_path << "failed !!!" << std::endl;
            continue;
        }
        // if use bin
        // utilities::read_file(image_path.c_str(), input_image);
        common::get_input_data_letterbox(src_image, input_image, input_size[0], input_size[1], true);

        std::vector<detection::Object> objects;

        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

        ret = middleware::push_input(input_image, &io_data, io_info);
        if (0 != ret)
        {
            printf("middleware::push_input error !!!\n");
            continue;
        }

        ret = AX_ENGINE_RunSync(handle, &io_data);
        if (0 != ret)
        {
            printf("AX_ENGINE_RunSync error !!!\n");
            continue;
        }

        auto src_h = src_image.rows;
        auto src_w = src_image.cols;

        post_process(io_info, &io_data, input_size, objects, src_h, src_w);
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double> >(end_time - start_time);
        double took_ms = time_used.count() * 1000.0f;
        std::cout << " infer time: " << took_ms << "ms" << std::endl;
        for (auto object : objects)
        {
            if (is_first)
            {
                fprintf(file_handle, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%.3f,%.3f,%.3f,%.3f], \"score\":%.6f}",
                        atoi(file_name_index.c_str()), coco80_to_coco91[object.label], detection::round_bit(object.rect.x, 3), detection::round_bit(object.rect.y, 3), detection::round_bit(object.rect.width, 3), detection::round_bit(object.rect.height, 3), detection::round_bit(object.prob, 5));
                is_first = false;
            }
            else
            {
                fprintf(file_handle, ",{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%.3f,%.3f,%.3f,%.3f], \"score\":%.6f}",
                        atoi(file_name_index.c_str()), coco80_to_coco91[object.label], detection::round_bit(object.rect.x, 3), detection::round_bit(object.rect.y, 3), detection::round_bit(object.rect.width, 3), detection::round_bit(object.rect.height, 3), detection::round_bit(object.prob, 5));
            }
        }

        count = count + 1;

        if (count > warmup_count)
        {
            total_time = total_time + took_ms;
        }
    }

    fprintf(file_handle, "]");
    fclose(file_handle);

    std::cout << "mean infer time: " << total_time / (count - warmup_count) << " ms" << std::endl;

    middleware::free_io(&io_data);
    return AX_ENGINE_DestroyHandle(handle);
}

int main(int argc, char* argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("images", 'i', "images dir", true, "");
    cmd.add<std::string>("save", 's', "save predict images dir", true, "");
    cmd.add<std::string>("size", 'g', "input_h, input_w", false, std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));
    cmd.parse_check(argc, argv);

    auto model_file = cmd.get<std::string>("model");
    auto images_dir = cmd.get<std::string>("images");
    auto save_dir = cmd.get<std::string>("save");

    auto model_file_flag = utilities::file_exist(model_file);
    auto image_file_flag = utilities::file_exist(images_dir);
    auto save_file_flag = utilities::file_exist(images_dir);

    if (!model_file_flag | !image_file_flag | !save_file_flag)
    {
        auto show_error = [](const std::string& kind, const std::string& value) {
            fprintf(stderr, "Input file %s(%s) is not exist, please check it.\n", kind.c_str(), value.c_str());
        };

        if (!model_file_flag)
        {
            show_error("model", model_file);
        }
        if (!image_file_flag)
        {
            show_error("images", images_dir);
        }
        if (!save_file_flag)
        {
            show_error("save", save_dir);
        }

        return -1;
    }

    auto input_size_string = cmd.get<std::string>("size");

    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};

    auto input_size_flag = utilities::parse_string(input_size_string, input_size);

    if (!input_size_flag)
    {
        auto show_error = [](const std::string& kind, const std::string& value) {
            fprintf(stderr, "Input %s(%s) is not allowed, please check it.\n", kind.c_str(), value.c_str());
        };

        show_error("size", input_size_string);

        return -1;
    }

    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "model file : %s\n", model_file.c_str());
    fprintf(stdout, "images dir : %s\n", images_dir.c_str());
    fprintf(stdout, "img_h, img_w : %d %d\n", input_size[0], input_size[1]);
    fprintf(stdout, "--------------------------------------\n");

    AX_SYS_Init();

    run_model(model_file, images_dir, input_size, save_dir);

    AX_ENGINE_Deinit();

    AX_SYS_Deinit();

    return 0;
}
