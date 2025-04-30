import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.dataset.h2r_dataset import H2RDataset

OmegaConf.register_new_resolver("eval", lambda expr: eval(expr))


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("../diffusion_policy", "config")
    ),
    config_name="my",
)
def test(cfg: OmegaConf):
    # 初始化类
    # cls = hydra.utils.get_class(cfg._target_)
    # new_cfg = {**cfg.task.dataset}
    new_cfg = cfg.task.dataset
    # print(new_cfg)
    # print(cfg.task.dataset)
    model = hydra.utils.instantiate(new_cfg)


if __name__ == "__main__":
    test()
