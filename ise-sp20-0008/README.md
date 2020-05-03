#Energy Alternatives

To run everything run the make command:
```
make docker-all
```
URL:
```
localhost:8080/EnergyAlternatives
```
To end the container:
```
make docker-stop
```

##Endpoints:

### /energy/svr/scatter/col/<arg1>/kernel/<arg2>/c/<arg3>
displays the scatter plot of the selected data (col) and its relationship to the other energy productions

### /energy/svr/plot/col/<arg1>/kernel/<arg2>/c/<arg3>
displays the accuracy of the predictions against the tested data over time
