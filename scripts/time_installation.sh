start=$SECONDS
conda env create -f environment.yml --name tmp_hylfm_env_for_timing_installation
duration=$(( SECONDS - start ))
echo duration: $duration s
conda env remove --name tmp_hylfm_env_for_timing_installation
# measure download speed:
curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python -
