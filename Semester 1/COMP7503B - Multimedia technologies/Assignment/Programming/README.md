# Preamble:

This assignment is composed of two parts: a) Programming Part, and b) Written Part.
The purpose of this assignment is to get you familiar with the concepts of multimedia
systems, through the realization of Smart City Use Case(s). In essence, you are required to
acquire appropriate data streams from https://data.gov.hk/, and based on these data to craft
up use cases and derive insights in relation to smart city initiatives. There is no fixed scope
nor limit for the smart city use cases, you are free to choose appropriate use cases as you see
fit.

The programming part requires you to implement an application that fetch data streams from
https://data.gov.hk/, data analysis and correlation, as well as dashboards to present insights
in relation to your selected smart city use cases. The written part, on the other hand, requires
you to write a report, in which you explain in details your smart city use cases, and
corresponding implementation details.

# Environment Setup:

**WSL2 needs to be installed on Windows**

1. Pull the docker images for Node-RED and MongoDB

```bash
docker pull nodered/node-red
docker pull mongo:latest
```

2. Run Node-RED Container

```bash
sudo mkdir /opt/node_red
sudo chmod 777 /opt/node_red
sudo docker run -itd -p 1880:1880 -v /opt/node_red:/data --name nodered nodered/node-red
```

Then access Node-RED at `http://localhost:1880`

3. Run MongoDB Container

```bash
sudo mkdir /opt/mongodb
sudo chmod 777 /opt/mongodb
sudo docker run -itd -p 27017:27017 -v /opt/mongodb:/data/db --name Mymongo mongo:latest
```

```bash
sudo docker exec -it Mymongo bash
mongosh

use admin
db.createUser({user: "admin", pwd: "admin", roles: [{role: "root", db: "admin"}]});
exit
```

4. Install the required Node-RED packages

```bash
sudo docker exec -it nodered bash
npm install node-red-dashboard # For the dashboard ui_chart which is not preinstalled
npm install node-red-contrib-mongodb3 
```

5. Import the Node-RED flow
6. Deploy the Node-RED flow
7. Access the dashboard at `http://localhost:1880/ui`