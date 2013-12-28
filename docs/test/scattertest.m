%This is a matlab script used for testing the functionality of the scatter plot.

%you will need to manually input the following fields: 
%files to load, and the data loaded. this script will not prompt for these.

%files
prefix = '/rri_disks/astraea/strother_lab/sslab/fjohnson/';
file2 = '2grp_10subj_3cls_2runs_fMRIresult.mat';
%file2 = '10subj_30cls_allRuns_3splits_020pc_NPAIRSJresult.mat';
file1 = '20subj_3cls_java_fMRIresult.mat';
%file1 = 'y246_NPAIRSJresult.mat';
%file2 = 'y246_5cond_NPAIRSJresult.mat';

loadedfile1 = load(strcat(prefix,file1));
loadedfile2 = load(strcat(prefix,file2));

%find common voxels
intersection = intersect(loadedfile1.st_coords,loadedfile2.st_coords);
lengthIntersection = length(intersection);

%extract data
%set 1
sprintf('Extracting from file %s', file1)

%data = loadedfile1.npairs_result.zscored_brainlv_avg;
%data = loadedfile1.boot_result.compare;
data = loadedfile1.brainlv;
ws = loadedfile1.st_win_size;
cv = input('input cv:');
lag = input('input lag:');
cords = length(data);
set1 = ones(1,length(intersection));


i = 1;
nextCord = 1;
for j = (lag:ws:cords)
    maskIndex = loadedfile1.st_coords(nextCord);
    for k = (1:lengthIntersection)
        if(maskIndex == intersection(k))
            set1(i) = data(j,cv);
            i = i + 1;
            break;
        end
   end
   nextCord = nextCord + 1;
end


%set 2
sprintf('Extracting from file %s', file2)
%data = loadedfile2.npairs_result.zscored_brainlv_avg;
%data = loadedfile2.npairs_result.cv_brainlv_avg;
data = loadedfile2.brainlv;
%data = loadedfile2.boot_result.compare;
ws = loadedfile2.st_win_size;
cv = input('input cv:');
lag = input('input lag:');
cords = length(data);
set2 = ones(1,length(intersection));

i = 1;
nextCord = 1;
for j = (lag:ws:cords)
   maskIndex = loadedfile2.st_coords(nextCord);
   for k = 1:lengthIntersection
       %if(loadedfile2.st_coords(nextCord) == intersection(k))
       if(maskIndex == intersection(k))
           set2(i) = data(j,cv);
           i = i + 1;
           break;
       end
   end
   nextCord = nextCord + 1;
end

%calculate pearsons correlation
     corrcoef(set1,set2)

%plot
     scatter(set1,set2)
