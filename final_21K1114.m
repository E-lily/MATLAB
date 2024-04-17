%"http://emodb.bilderbar.info/download/download.zip"からEmo-DBをダウンロードする。
%Emo-DBの中にdownload.zipをインストールし、フォルダのパスは適宜変更する。


dataFolder = tempdir;
dataset = fullfile(dataFolder, "Emo-DB");

%{
% フォルダのパスとzipファイル名
folderPath = 'C:\Users\admin\Desktop\Documents\3年秋学期\音声情報処理\Emo-DB';
zipFileName = 'download.zip';

% 解凍先のフォルダ
outputFolder = folderPath;

% 解凍
unzip(fullfile(folderPath, zipFileName), outputFolder);
%}

% フォルダのパス
audioFolderPath = 'C:\Users\admin\Desktop\Documents\3年秋学期\音声情報処理\Emo-DB\wav';

ads = audioDatastore(audioFolderPath, 'IncludeSubfolders', true);



filepaths = ads.Files;
emotionCodes = cellfun(@(x)x(end-5),filepaths,UniformOutput=false);
emotions = replace(emotionCodes,["W","L","E","A","F","T","N"], ...
    ["Anger","Boredom","Disgust","Anxiety/Fear","Happiness","Sadness","Neutral"]);

speakerCodes = cellfun(@(x)x(end-10:end-9),filepaths,UniformOutput=false);
labelTable = cell2table([speakerCodes,emotions],VariableNames=["Speaker","Emotion"]);
labelTable.Emotion = categorical(labelTable.Emotion);
labelTable.Speaker = categorical(labelTable.Speaker);
summary(labelTable)



ads.Labels = labelTable;



downloadFolder = matlab.internal.examples.downloadSupportFile("audio","SpeechEmotionRecognition.zip");
dataFolder = tempdir;
unzip(downloadFolder,dataFolder)
netFolder = fullfile(dataFolder,"SpeechEmotionRecognition");
load(fullfile(netFolder,"network_Audio_SER.mat"));

fs = afe.SampleRate;

speaker = categorical("08");
emotion = categorical("Anger");

adsSubset = subset(ads,ads.Labels.Speaker==speaker & ads.Labels.Emotion==emotion);

audio = read(adsSubset);
sound(audio,fs)




features = (extract(afe,audio))';

featuresNormalized = (features - normalizers.Mean)./normalizers.StandardDeviation;

numOverlap = 10;
featureSequences = HelperFeatureVector2Sequence(featuresNormalized,20,numOverlap);


YPred = double(predict(net,featureSequences));

average = "mode";
switch average
    case "mean"
        probs = mean(YPred,1);
    case "median"
        probs = median(YPred,1);
    case "mode"
        probs = mode(YPred,1);
end

pie(probs./sum(probs))
%プロット
legend(string(net.Layers(end).Classes),Location="eastoutside");



% SpeakerとEmotionの一覧を取得
uniqueSpeakers = unique(ads.Labels.Speaker);
uniqueEmotions = unique(ads.Labels.Emotion);

% 正答率を保存
accuracyResults = zeros(length(uniqueSpeakers), length(uniqueEmotions));

% Speakerごとにループ
for i = 1:length(uniqueSpeakers)
    currentSpeaker = uniqueSpeakers(i);
    
    % Emotionごとにループ
    for j = 1:length(uniqueEmotions)
        currentEmotion = uniqueEmotions(j);
        
        adsSubset = subset(ads, ads.Labels.Speaker == currentSpeaker & ads.Labels.Emotion == currentEmotion);

        if hasdata(adsSubset)
            audio = read(adsSubset);

            features = (extract(afe, audio))';

            featuresNormalized = (features - normalizers.Mean) ./ normalizers.StandardDeviation;

            numOverlap = 10;
            featureSequences = HelperFeatureVector2Sequence(featuresNormalized, 20, numOverlap);

            YPred = double(predict(net, featureSequences));

            average = "mode";
            switch average
                case "mean"
                    probs = mean(YPred, 1);
                case "median"
                    probs = median(YPred, 1);
                case "mode"
                    probs = mode(YPred, 1);
            end

            %パーセンテージを計算
            percentage = probs ./ sum(probs) * 100;

            %結果を保存
            accuracyResults(i, j) = percentage(find(currentEmotion == uniqueEmotions));
        end
    end
end

%プロット
figure;
bar(accuracyResults, 'stacked');
xlabel('Speaker');
ylabel('Accuracy (%)');
legend(uniqueEmotions, 'Location', 'Best');
title('発話者ごとの感情の正答率');















function [sequences,sequencePerFile] = HelperFeatureVector2Sequence(features,featureVectorsPerSequence,featureVectorOverlap)
    if featureVectorsPerSequence <= featureVectorOverlap
        error("The number of overlapping feature vectors must be less than the number of feature vectors per sequence.")
    end

    if ~iscell(features)
        features = {features};
    end
    hopLength = featureVectorsPerSequence - featureVectorOverlap;
    idx1 = 1;
    sequences = {};
    sequencePerFile = cell(numel(features),1);
    for ii = 1:numel(features)
        sequencePerFile{ii} = floor((size(features{ii},2) - featureVectorsPerSequence)/hopLength) + 1;
        idx2 = 1;
        for j = 1:sequencePerFile{ii}
            sequences{idx1,1} = features{ii}(:,idx2:idx2 + featureVectorsPerSequence - 1); %#ok<AGROW>
            idx1 = idx1 + 1;
            idx2 = idx2 + hopLength;
        end
    end
end

