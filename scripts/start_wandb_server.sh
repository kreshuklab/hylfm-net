if [ $1 == "--update" ];then
  echo update
  docker pull wandb/local
  docker stop wandb-local
fi

docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
