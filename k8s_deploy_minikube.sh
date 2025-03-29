minikube start
eval $(minikube docker-env)
docker compose build
kubectl apply -f k8s-manifest-minikube.yml
minikube dashboard


#export DOCKER_HOST=unix:///var/run/docker.sock