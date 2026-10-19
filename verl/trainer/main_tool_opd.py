"""Entry point for Tool-OPD training."""

import hydra
import ray
from omegaconf import OmegaConf, open_dict


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_tool_opd(config)


def run_tool_opd(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env = OmegaConf.merge(default_runtime_env, ray_init_kwargs.get("runtime_env", {}))
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(ToolOPDTaskRunner.remote().run.remote(config))


@ray.remote(num_cpus=1)
class ToolOPDTaskRunner:
    def run(self, config):
        from verl.utils.fs import copy_to_local

        tool_cfg = config.algorithm.get("tool_opd", None)
        if tool_cfg is None:
            raise ValueError("Missing +algorithm.tool_opd configuration")
        teacher_model_path = tool_cfg.get("teacher_model_path", None)
        if not teacher_model_path:
            raise ValueError("algorithm.tool_opd.teacher_model_path is required")

        teacher_config = OmegaConf.create(OmegaConf.to_container(config.actor_rollout_ref, resolve=True))
        teacher_config.model.path = teacher_model_path
        teacher_config.rollout.gpu_memory_utilization = tool_cfg.get(
            "teacher_gpu_memory_utilization", teacher_config.rollout.gpu_memory_utilization
        )
        teacher_config.rollout.mode = "sync"
        teacher_config.rollout.n = 1
        teacher_config.rollout.response_length = tool_cfg.get("hindsight_response_length", 256)
        teacher_prompt_length = max(
            tool_cfg.get("teacher_max_prompt_length", config.data.max_prompt_length),
            tool_cfg.get("hindsight_max_prompt_length", config.data.max_prompt_length),
        )
        teacher_config.rollout.prompt_length = teacher_prompt_length
        teacher_config.rollout.max_model_len = teacher_prompt_length + teacher_config.rollout.response_length
        with open_dict(config):
            config.teacher = teacher_config
        with open_dict(config.actor_rollout_ref.actor):
            config.actor_rollout_ref.actor.use_tool_opd_loss = True
            config.actor_rollout_ref.actor.tool_opd_loss_coef = tool_cfg.get("loss_coef", 0.01)
            config.actor_rollout_ref.actor.tool_opd_max_log_ratio = tool_cfg.get("max_log_ratio", 5.0)
            config.actor_rollout_ref.actor.use_kl_loss = False
            config.actor_rollout_ref.actor.use_sdl_loss = False
            config.actor_rollout_ref.actor.use_sdar_loss = False
            config.actor_rollout_ref.actor.entropy_coeff = 0.0

        student_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        teacher_path = copy_to_local(
            teacher_model_path,
            use_shm=teacher_config.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs
        from verl.utils import hf_processor, hf_tokenizer

        envs, val_envs = make_envs(config)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(student_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(student_path, trust_remote_code=trust_remote_code, use_fast=True)
        teacher_tokenizer = hf_tokenizer(teacher_path, trust_remote_code=trust_remote_code)

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Teacher: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Teacher: global_pool_id,
            Role.Critic: global_pool_id,
        }
        if tool_cfg.get("separate_teacher_pool", False):
            teacher_pool_id = "teacher_pool"
            teacher_gpus = tool_cfg.get("teacher_n_gpus_per_node", config.trainer.n_gpus_per_node)
            teacher_nodes = tool_cfg.get("teacher_nnodes", config.trainer.nnodes)
            resource_pool_spec[teacher_pool_id] = [teacher_gpus] * teacher_nodes
            mapping[Role.Teacher] = teacher_pool_id

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        from agent_system.reward_manager import EpisodeRewardManager
        from agent_system.multi_turn_rollout import TrajectoryCollector
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.tool_opd_ray_trainer import ToolOPDRayTrainer
        from verl.utils.dataset.rl_dataset import collate_fn

        reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = ToolOPDRayTrainer(
            config=config,
            tokenizer=tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
