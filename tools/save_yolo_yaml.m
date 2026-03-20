function save_yolo_yaml(out_base, class_names, yaml_name)

% save_yolo_yaml - Create a YAML configuration file for YOLO training
%
% This function creates a dataset configuration file in YAML format
% compatible with YOLO training pipelines.
%
% The generated YAML file contains:
%   - the absolute dataset path
%   - the relative paths to the train, validation, and test image folders
%   - the number of classes
%   - the mapping from class IDs to class names
%
% This step is required because YOLO training expects a YAML file
% describing the dataset structure and class labels.
%
% Input arguments:
%   out_base     - Base directory of the YOLO-formatted dataset
%
%   class_names  - Class names to be written to the YAML file.
%                  It may be provided as:
%                    * cell array of char
%                    * string array
%                    * char array
%
%   yaml_name    - Name of the output YAML file.
%                  If not provided or empty, the default name
%                  'config.yaml' is used.
%
% This function was developed by Prof. Eliezer Soares Flores
% (Google Scholar Profile: https://scholar.google.com.br/citations?user=3jNjADcAAAAJ&hl=pt-BR&oi=ao).
%
% If you have any questions, please feel free to contact me at:
% eliezersflores@gmail.com


    if nargin < 3 || isempty(yaml_name)
        yaml_name = 'config.yaml';
    end

    % Save the YAML file in the current working directory
    yaml_path = fullfile(pwd, yaml_name);

    % Get the absolute dataset path
    dataset_abs = char(java.io.File(out_base).getCanonicalPath());
    dataset_abs = strrep(dataset_abs, '\', '/');

    % Normalize class_names to a cell array of char
    if isstring(class_names)
        class_names = cellstr(class_names);
    elseif ischar(class_names)
        class_names = cellstr(class_names);
    elseif ~iscell(class_names)
        error('class_names must be a cell array, string array, or char array.');
    end

    fid = fopen(yaml_path, 'w');
    if fid == -1
        error('Unable to create the YAML file at: %s', yaml_path);
    end

    fprintf(fid, '# Paths\n');
    fprintf(fid, 'path: %s\n\n', dataset_abs);

    fprintf(fid, 'train: images/train\n');
    fprintf(fid, 'val: images/val\n');
    fprintf(fid, 'test: images/test\n\n');

    fprintf(fid, '# Class Labels\n');
    fprintf(fid, 'nc: %d\n', numel(class_names));
    fprintf(fid, 'names:\n');

    for i = 1:numel(class_names)
        fprintf(fid, '  %d: %s\n', i-1, class_names{i});
    end

    fclose(fid);

    fprintf('\nYAML file saved to: %s\n', yaml_path);

end
