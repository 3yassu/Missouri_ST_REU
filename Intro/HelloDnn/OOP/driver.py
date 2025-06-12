#driver.py
from executive import Executive

def main():
	runner = Executive()
	runner.train(0)
	runner.print()
main()
