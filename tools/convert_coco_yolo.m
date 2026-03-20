% =========================================================
% COCO-TO-YOLO CONVERSION SCRIPT
% =========================================================
% This script converts the simulated CSRD dataset from COCO
% format to YOLO format.
%
% It reads COCO annotations and STFT tensors, exports the tensors
% as PNG images, converts the bounding boxes to YOLO format, and
% organizes the output into train/val/test folders.
%
% This step is required because YOLO training expects image files
% and label files in its own annotation format rather than the
% original COCO-style representation.
%
% This script was developed by Prof. Eliezer Soares Flores
% (Google Scholar Profile: https://scholar.google.com.br/citations?user=3jNjADcAAAAJ&hl=pt-BR&oi=ao).
%
% If you have any questions, please feel free to contact me at
% eliezersflores@gmail.com
% =========================================================

clear; close all; clc;

% =========================================================
% CONFIGURATION
% =========================================================

window_name = 'hamming';   % 'hamming', 'hann', or 'blackman'
granularity = 'family9';   % 'binary', 'family9', or 'original100'
resize_mode = 'offline';   % 'offline' or 'yolo'
target_size = 640;         % used when resize_mode = 'offline'

splits = {'train', 'val', 'test'};

% Define a resize tag that includes the target size when applicable
switch resize_mode
    case 'offline'
        resize_tag = sprintf('%s%d', resize_mode, target_size);
    case 'yolo'
        resize_tag = resize_mode;
    otherwise
        error('Invalid resize_mode: %s', resize_mode);
end

out_base = ['../data/CSRD2025/yolo_' window_name '_' granularity '_' resize_tag];

for s = 1:numel(splits)

    split = splits{s};

    fprintf('\n=========================================================\n');
    fprintf('PROCESSING SPLIT: %s\n', split);
    fprintf('=========================================================\n');

    path_mat = ['../data/CSRD2025/stft/' window_name];
    coco_json = ['../data/CSRD2025/coco_annotations_' window_name '_' split '.json'];
    
    out_images = [out_base '/images/' split];
    out_labels = [out_base '/labels/' split];
    
    if ~exist(out_images, 'dir')
        mkdir(out_images);
    end
    
    if ~exist(out_labels, 'dir')
        mkdir(out_labels);
    end
    
    % =========================================================
    % READ COCO JSON FILE
    % =========================================================
    fprintf('Reading COCO JSON file: %s\n', coco_json);
    json_text = fileread(coco_json);
    coco = jsondecode(json_text);
    
    images_info = coco.images;
    annotations = coco.annotations;
    categories  = coco.categories;
    
    fprintf('Number of images in split %s: %d\n', split, numel(images_info));
    fprintf('Number of annotations in split %s: %d\n', split, numel(annotations));
    fprintf('Number of original categories: %d\n', numel(categories));
    
    % =========================================================
    % BUILD MAP: COCO category_id -> category name
    % =========================================================
    cat_id_to_name = containers.Map('KeyType', 'double', 'ValueType', 'char');
    
    for k = 1:numel(categories)
        cat_id_to_name(categories(k).id) = categories(k).name;
    end
    
    % =========================================================
    % DEFINE CLASS MAPPING ACCORDING TO THE CHOSEN GRANULARITY
    % =========================================================
    
    [class_map, class_names] = build_class_map(categories, granularity);
    
    num_classes = numel(class_names);
    class_count = zeros(1, num_classes);
    
    fprintf('Selected granularity: %s\n', granularity);
    fprintf('Number of YOLO classes: %d\n', num_classes);
    
    % Display the mapping from original COCO categories to YOLO classes
    cat_ids = [categories.id];
    [~, idx_sort] = sort(cat_ids);
    categories_sorted = categories(idx_sort);
    
    for k = 1:numel(categories_sorted)
        cat_name = categories_sorted(k).name;
        cat_id   = categories_sorted(k).id;
        class_id = class_map(cat_name);
    
        fprintf('COCO category_id=%d (%s) -> YOLO class_id=%d (%s)\n', ...
            cat_id, cat_name, class_id, class_names{class_id+1});
    
    end
    
    % =========================================================
    % GROUP ANNOTATIONS BY image_id
    % =========================================================
    annotation_image_ids = [annotations.image_id];
    
    % =========================================================
    % MAIN LOOP
    % =========================================================
    for i = 1:numel(images_info)
    
        img_id = images_info(i).id;
        mat_name = images_info(i).file_name;
        mat_path = fullfile(path_mat, mat_name);
    
        % -----------------------------------------------------
        % LOAD THE .MAT FILE
        % -----------------------------------------------------
        data = load(mat_path);
    
        % Use only channel 1
        img = data.stftTensor(:,:,1);
    
        % Convert to double and normalize to [0, 255]
        img = double(img);
    
        % In some cases, the image may contain negative values
        % or an arbitrary dynamic range
        img_min = min(img(:));
        img_max = max(img(:));
    
        if img_max > img_min
            img_norm = (img - img_min) / (img_max - img_min);
        else
            img_norm = zeros(size(img));
        end
    
        img_uint8 = uint8(255 * img_norm);
        
        % Store original dimensions
        orig_h = size(img_uint8, 1);
        orig_w = size(img_uint8, 2);
        
        % -----------------------------------------------------
        % OPTIONAL RESIZING
        % offline: resize to target_size x target_size before saving
        % yolo: keep the original resolution; YOLO will apply
        %       letterboxing during training
        % -----------------------------------------------------
    
        switch resize_mode
            case 'offline'
                new_h = target_size;
                new_w = target_size;
                img_uint8 = imresize(img_uint8, [new_h, new_w]);
        
            case 'yolo'
                new_h = orig_h;
                new_w = orig_w;
        
            otherwise
                error('Invalid resize_mode: %s', resize_mode);
        end
        
        % -----------------------------------------------------
        % SAVE PNG IMAGE
        % -----------------------------------------------------
        [~, base_name, ~] = fileparts(mat_name);
        png_name = [base_name '.png'];
        png_path = fullfile(out_images, png_name);
    
        imwrite(img_uint8, png_path);
    
        % -----------------------------------------------------
        % FIND ANNOTATIONS ASSOCIATED WITH THIS IMAGE
        % -----------------------------------------------------
        idx_ann = find(annotation_image_ids == img_id);
    
        label_path = fullfile(out_labels, [base_name '.txt']);
        fid = fopen(label_path, 'w');
    
        img_h = size(img_uint8, 1);
        img_w = size(img_uint8, 2);
    
        scale_x = new_w / orig_w;
        scale_y = new_h / orig_h;
    
        for j = 1:numel(idx_ann)
            ann = annotations(idx_ann(j));
    
            % COCO bounding box format: [x, y, width, height]
            bbox = ann.bbox;
    
            x = bbox(1) * scale_x;
            y = bbox(2) * scale_y;
            w = bbox(3) * scale_x;
            h = bbox(4) * scale_y;
    
            % Enforce minimum box size
            w = max(w, 1);
            h = max(h, 1);
    
            % Convert COCO -> YOLO
            x_center = x + w/2;
            y_center = y + h/2;
    
            x_center_norm = x_center / img_w;
            y_center_norm = y_center / img_h;
            w_norm = w / img_w;
            h_norm = h / img_h;
    
            % Clamp values for safety
            x_center_norm = min(max(x_center_norm, 0), 1);
            y_center_norm = min(max(y_center_norm, 0), 1);
            w_norm        = min(max(w_norm, 0), 1);
            h_norm        = min(max(h_norm, 0), 1);
    
            coco_category_id = ann.category_id;
            category_name = cat_id_to_name(coco_category_id);
    
            class_id = class_map(category_name);
            class_count(class_id + 1) = class_count(class_id + 1) + 1;
    
    
            fprintf(fid, '%d %.6f %.6f %.6f %.6f\n', ...
                class_id, x_center_norm, y_center_norm, w_norm, h_norm);
        end
    
        fclose(fid);
    
        if mod(i,100) == 0 || i == numel(images_info)
            fprintf('Processed %d/%d images from split %s\n', i, numel(images_info), split);
        end
    end
    
    fprintf('\nFinal class distribution:\n');
    for k = 1:numel(class_names)
        fprintf('class_id=%d | %s | n=%d\n', k-1, class_names{k}, class_count(k));
    end
    
    fprintf('\nConversion completed for split %s.\n', split);
    fprintf('Images saved to: %s\n', out_images);
    fprintf('Labels saved to: %s\n', out_labels);

end

yaml_name = sprintf('config_yolo_%s_%s_%s.yaml', ...
    window_name, granularity, resize_tag);
    
save_yolo_yaml(out_base, class_names, yaml_name);
