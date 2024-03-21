/*
 * Jenkinsfile to pull the source code from git, build a docker image
 * using a Dockerfile and push that image to a registry.
 */


pipeline {

  agent any

  environment {
    /*
     * edit the following variables
     */
    
    // the name of your docker image
    // set to 'datalabauth/projectname' for Dockerhub
    // or just to 'projectname' if you're using our private registry
    dockertag = 'ai4netmon-dashboard'

    // the registry name
    // set to 'https://registry.hub.docker.com' for DockerHub
    // or to 'https://registry.csd.auth.gr' for our private registry
    registry = 'https://registry.csd.auth.gr'

    // the credentials to the above registry as stored in Jenkins
    // set to 'dockerhub' for DockerHub 
    // or 'datalab-registry' for our private registry
    registry_credentials = 'datalab-registry'
  }

  // you probably don't need to edit anything below this line
  stages {
           
    stage('Checkout the source code') {
      steps {
        checkout scm
      }
    }

    stage('Build') {
      steps {
        script {
          image = docker.build("$dockertag")
        }
      }
    }

    stage('Push') {
      steps {
        script {
          docker.withRegistry(registry, registry_credentials) {
            image.push()
          }
        }
      }
    }
  }
}


