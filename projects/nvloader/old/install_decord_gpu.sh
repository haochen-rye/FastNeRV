# decord
git clone --recursive https://github.com/dmlc/decord
cd decord
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release  -DFFMPEG_DIR=/data/home/haochen1/ffmpeg_build
make -j32

# -DFFMPEG_DIR=/data/home/haochen1/local/FFmpeg-n4.4.2
#  -DFFMPEG_DIR=/data/home/haochen1/local/ffmpeg-n4.4-latest-linux64-gpl-4.4/bin/ffmpeg
#  -CUDA_TOOLKIT_ROOT_DIR\lib64=/lib/x86_64-linux-gnu/libnvcuvid.so -DFFMPEG_DIR=/data/home/haochen1/local/ffmpeg-n4.4-latest-linux64-gpl-4.4/bin/ffmpeg

pwd=$PWD
echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bash_profile
source ~/.bash_profile



# ffmpeg
cd ~/local/FFmpeg-n4.4.2
# ccap=80
./configure --prefix=$HOME/ffmpeg_build --disable-static --enable-shared --enable-gpl --enable-nonfree --disable-libx264 \
    --enable-decoder=h264 --enable-decoder=h264_cuvid --enable-encoder=libx264 --enable-encoder=h264_nvenc  \
    # --nvccflags="-gencode arch=compute_${ccap},code=sm_${ccap} -O2" 
make -j16
make install
export PATH=$HOME/ffmpeg_build/bin:$PATH


ccap=80
./configure \
  --prefix='/data/home/haochen1/local/ffmpeg_build' \
  --extra-cflags='-I/usr/local/cuda/include' \
  --extra-ldflags='-L/usr/local/cuda/lib64' \
  --nvccflags="-gencode arch=compute_${ccap},code=sm_${ccap} -O2" \
  --disable-doc \
  --disable-static \
  --disable-bsfs \
  --disable-decoders \
  --disable-encoders \
  --disable-filters \
  --disable-demuxers \
  --disable-devices \
  --disable-muxers \
  --disable-parsers \
  --disable-postproc \
  --disable-protocols \
  --enable-decoder=aac \
  --enable-decoder=h264 \
  --enable-decoder=h264_cuvid \
  --enable-decoder=rawvideo \
  --enable-indev=lavfi \
  --enable-encoder=libx264 \
  --enable-encoder=h264_nvenc \
  --enable-demuxer=mov \
  --enable-muxer=mp4 \
  --enable-filter=scale \
  --enable-filter=testsrc2 \
  --enable-protocol=file \
  --enable-protocol=https \
  --disable-gnutls \
  --enable-shared \
  --enable-gpl \
  --enable-nonfree \
  --enable-cuda-nvcc \
  --disable-libx264 \
  --enable-nvenc \
  --enable-cuvid \
  --enable-nvdec


./configure --prefix=/data/home/haochen1/local
make
make check
sudo make install


# extract video
ffmpeg -i /data/home/haochen1/data/kinetics_vid/k400_train_libx264_veryfast_crf25_256_F8/--uWuXty1jw_000027_000037.mp4 -vf "select=not(mod(n\,6)),setpts=N/FRAME_RATE/TB" -vframes N_frames -q:v 2 -vsync vfr output.mp4
output_%03d.png


ffmpeg -i input2.mp4 -vf "select=not(mod(n\,15))" -q:v 2 -vsync vfr -y output2.mp4
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 output2.mp4

ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 /data/home/haochen1/data/kinetics_vid_evenly/k400_train_libx264_superfast_crf45_256_F8/0FaXLfqcphQ_000015_000025.mp4
/data/home/haochen1/data/kinetics_vid_evenly/k400_train_libx264_veryfast_crf25_256_F8/--mQfG6Wu48_000028_000038.mp4


libx264  -tune fastdecode crf 50
av1 -svtav1-params fast-decode=3 crf 60


