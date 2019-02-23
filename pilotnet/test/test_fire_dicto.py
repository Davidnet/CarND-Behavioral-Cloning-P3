import dicto as dc 
import fire

@dc.fire_options("data-augmentation.yml")
def main(others, params):
    import pdb; pdb.set_trace()
    print(params)
    print(others)


if __name__ == "__main__":
    fire.Fire(main)