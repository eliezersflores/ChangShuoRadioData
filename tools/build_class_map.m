function [class_map, class_names] = build_class_map(categories, granularity)

% build_class_map - Create the mapping from original dataset classes to YOLO classes
%
% This function defines how the original class names in the dataset
% are mapped to the output classes used for YOLO training.
%
% Supported granularity levels:
%   - 'binary'      : all classes are mapped to a single class ('signal')
%   - 'family9'     : original classes are grouped into 9 modulation families
%   - 'original100' : all original classes are preserved as independent classes
%
% Input arguments:
%   categories   - Struct array of dataset categories. Each element is
%                  expected to contain at least the fields:
%                     .id   : category identifier
%                     .name : category name
%
%   granularity  - String specifying the desired class granularity.
%                  Supported values:
%                     'binary', 'family9', 'original100'
%
% Output arguments:
%   class_map    - containers.Map that associates each original class
%                  name with its corresponding YOLO class ID
%
%   class_names  - Cell array containing the YOLO class names in the order
%                  expected by YOLO (0-based indexing in class_map, 1-based
%                  indexing in MATLAB cell arrays)
%
% Notes:
%   - YOLO uses 0-based class indexing in label files.
%
% This function was developed by Prof. Eliezer Soares Flores
% (Google Scholar Profile: https://scholar.google.com.br/citations?user=3jNjADcAAAAJ&hl=pt-BR&oi=ao).
%
% If you have any questions, please feel free to contact me at:
% eliezersflores@gmail.com

    class_map = containers.Map('KeyType', 'char', 'ValueType', 'double');

    switch lower(granularity)

        case 'binary'
            % YOLO does not require an explicit background class
            class_names = {'signal'};

            % Map all original classes to a single class
            for k = 1:numel(categories)
                class_map(categories(k).name) = 0;
            end

        case 'family9'

            % Define the 9 output modulation families
            class_names = { ...
                'AM', ...
                'FM', ...
                'PM', ...
                'ASK', ...
                'FSK', ...
                'CPM', ...
                'PSK', ...
                'QAM', ...
                'APSK'};

            % -------------------------
            % AM family
            % -------------------------
            class_map('SSBAM')   = 0;
            class_map('DSBAM')   = 0;
            class_map('DSBSCAM') = 0;
            class_map('VSBAM')   = 0;

            % -------------------------
            % FM family
            % -------------------------
            class_map('FM')      = 1;

            % -------------------------
            % PM family
            % -------------------------
            class_map('PM')      = 2;

            % -------------------------
            % ASK family
            % -------------------------
            ask_list = {'2-OOK','4-ASK','8-ASK','16-ASK','32-ASK','64-ASK'};
            for k = 1:numel(ask_list)
                class_map(ask_list{k}) = 3;
            end

            % -------------------------
            % FSK family
            % -------------------------
            fsk_list = {'2-FSK','4-FSK','8-FSK','4-GFSK','8-GFSK'};
            for k = 1:numel(fsk_list)
                class_map(fsk_list{k}) = 4;
            end

            % -------------------------
            % CPM family
            % -------------------------
            cpm_list = {'2-GMSK','2-MSK','4-CPFSK','8-CPFSK'};
            for k = 1:numel(cpm_list)
                class_map(cpm_list{k}) = 5;
            end

            % -------------------------
            % PSK family
            % -------------------------
            psk_list = { ...
                '2-PSK','4-PSK','8-PSK','16-PSK','32-PSK','64-PSK', ...
                '4-OQPSK', ...
                '2-PSK-OFDM','4-PSK-OFDM','8-PSK-OFDM','16-PSK-OFDM','32-PSK-OFDM','64-PSK-OFDM', ...
                '2-PSK-SCFDMA','4-PSK-SCFDMA','8-PSK-SCFDMA','16-PSK-SCFDMA','32-PSK-SCFDMA','64-PSK-SCFDMA', ...
                '2-PSK-OTFS','4-PSK-OTFS','8-PSK-OTFS','16-PSK-OTFS','32-PSK-OTFS','64-PSK-OTFS'};
            for k = 1:numel(psk_list)
                class_map(psk_list{k}) = 6;
            end

            % -------------------------
            % QAM family
            % -------------------------
            qam_list = { ...
                '8-QAM','16-QAM','32-QAM','64-QAM','128-QAM','256-QAM','512-QAM','1024-QAM','2048-QAM','4096-QAM', ...
                '8-QAM-OFDM','16-QAM-OFDM','32-QAM-OFDM','64-QAM-OFDM','128-QAM-OFDM','256-QAM-OFDM','512-QAM-OFDM','1024-QAM-OFDM','2048-QAM-OFDM','4096-QAM-OFDM', ...
                '8-QAM-SCFDMA','16-QAM-SCFDMA','32-QAM-SCFDMA','64-QAM-SCFDMA','128-QAM-SCFDMA','256-QAM-SCFDMA','512-QAM-SCFDMA','1024-QAM-SCFDMA','2048-QAM-SCFDMA','4096-QAM-SCFDMA', ...
                '8-QAM-OTFS','16-QAM-OTFS','32-QAM-OTFS','64-QAM-OTFS','128-QAM-OTFS','256-QAM-OTFS','512-QAM-OTFS','1024-QAM-OTFS','2048-QAM-OTFS','4096-QAM-OTFS', ...
                '16-Mill88QAM','32-Mill88QAM','64-Mill88QAM','256-Mill88QAM'};
            for k = 1:numel(qam_list)
                class_map(qam_list{k}) = 7;
            end

            % -------------------------
            % APSK family
            % -------------------------
            apsk_list = { ...
                '16-APSK','32-APSK','64-APSK','128-APSK','256-APSK', ...
                '16-DVBSAPSK','32-DVBSAPSK','64-DVBSAPSK','128-DVBSAPSK','256-DVBSAPSK'};
            for k = 1:numel(apsk_list)
                class_map(apsk_list{k}) = 8;
            end

        case 'original100'

            % Keep all original classes as independent output classes
            cat_ids = [categories.id];
            [~, idx_sort] = sort(cat_ids);
            categories_sorted = categories(idx_sort);

            class_names = cell(1, numel(categories_sorted));

            for k = 1:numel(categories_sorted)
                cname = categories_sorted(k).name;
                class_id = k - 1;  % 0-based indexing for YOLO
                class_map(cname) = class_id;
                class_names{class_id + 1} = cname;
            end

        otherwise
            error('Invalid granularity "%s". Use binary, family9, or original100.', granularity);
    end
end
