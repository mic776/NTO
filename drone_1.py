from time import sleep
import pioneer_sdk2

pioneer = pioneer_sdk2.Pioneer()

def wait_point():
    while not pioneer.point_reached():
        sleep(0.05)

pioneer.go_to_local_point(-0.2, 0, 0.3)
wait_point()
print("1")

pioneer.go_to_local_point(-0.2, -1, 0.3)
wait_point()
print("2")

pioneer.go_to_local_point(0.2, -1, 0.3)
wait_point()
print("3")

pioneer.go_to_local_point(0.2, 0, 0.3)
wait_point()
print("4")

pioneer.go_to_local_point(0, 0, 0)
wait_point()
print("home")
