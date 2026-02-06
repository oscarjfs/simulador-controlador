from simulador_controlador import SimuladorControlador
import logging


def main():
    # Configurar logging a nivel INFO
    logging.basicConfig(level=logging.INFO)
    simulador = SimuladorControlador()
    simulador.ejecutar()


if __name__ == "__main__":
    main()
