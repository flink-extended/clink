Clink provides dockerfiles in this folder to create Docker images that provide
the environment required to build and execute Clink programs. This README
contains guidelines of how to build Clink Docker images and push them to Docker
Hub.

**When should Clink Docker images on Docker Hub be updated?**

What Clink images provides is the environment to execute Clink programs, not
Clink itself, so Clink images on Docker Hub need to be updated only when the
environment required for Clink changes. It is worth noting that Clink
dockerfiles also cache prebuilt TFRT in the images, thus changing the commit id
of the TFRT submodule of this repository also means that Clink Docker images
should be updated.

## Prerequisites

- Make sure you have installed [Docker](https://docs.docker.com/engine/install/)
  on your system.
- Make sure that you have write permission to `flinkextended/clink`, which means
  one of the following.
  - You are authenticated with your Docker ID, and that your Docker ID has
    access to `flinkextended/clink`, or
  - You have a `flinkextended/clink`'s access token, so that you can directly
    login to `flinkextended`.
  - If you do not have such permissions, please contact the reviewers of your PR
    and request access to `flinkextended` from them. They are supposed to have
    been familiar with Clink and know where to get such access.

## Docker Image Publication Guideline

1. Build the Docker image locally. From this directory:

   ```sh
   $ docker build -t clink:<docker-image-tag> -f <dockerfile-filename> ..
   ```

2. Create the Docker image to be published from locally built image.

   ```sh
   $ docker tag clink:<docker-image-tag> flinkextended/clink:<docker-image-tag>
   ```

3. Log in to the Docker Hub account with write permission.

   ```sh
   $ docker login -u <your-docker-hub-id>
   ```

4. Push the Docker image to Docker Hub.

   ```sh
   $ docker push flinkextended/clink:<docker-image-tag>
   ```
