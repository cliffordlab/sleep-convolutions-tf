
from subprocess import check_output
from . import tools as tl


class CloudProcess:

    cmd_base = None

    def __init__(self):
        self.command = None
        pass


class MLEngineProcess:

    cloud_cmd_base = ('gcloud', 'ml-engine', 'jobs', 'submit', 'training')
    local_cmd_base = ('gcloud', 'ml-engine', 'local', 'train')

    default_local_package_args = {
        'module_name': 'trainer.task',
        'package_path': 'trainer/',
        'config': None,
    }

    default_cloud_package_args = {
        'module_name': 'trainer.task',
        'package_path': 'trainer/',
        'runtime_version': '1.2',
        'job_dir': './logs/test',
        'region': 'us-east1',
        # 'scale_tier': 'BASIC_GPU',
        'config': None,
    }

    def __init__(self, job_name, **kwargs):
        self.job_name = job_name
        self.job_kwargs = kwargs
        if 'local' in kwargs and kwargs.pop('local'):
            self.cmd_base = self.local_cmd_base
            self.package_kwargs = tl.pop_params_or_defaults(kwargs,
                    self.default_local_package_args)
            self.package_kwargs.pop('config')
        else:
            self.cmd_base = self.cloud_cmd_base + (self.job_name,)
            self.package_kwargs = tl.pop_params_or_defaults(kwargs,
                    self.default_cloud_package_args)

    @property
    def job_name(self):
        return self._job_name

    @job_name.setter
    def job_name(self, v):
        self._job_name = v.replace(' ', '_').replace('-', '_')

    def build_command(self):
        # base:  gcloud ml-engine ..
        cmd_list = self.cmd_base

        # package cmds:  --package-name trainer.task ..
        for k, v in self.package_kwargs.items():
            if v is not None:
                cmd_list += (
                    "--{}".format(k.replace('_', '-')),
                    "{}".format(v)
                )

        cmd_list += ('--',)

        # job infos:  --test-data 'd1' 'd2' --opt rmsprop ..
        for k, v in self.job_kwargs.items():
            if v is not None:
                if isinstance(v, list):
                    v = tuple(v)
                if isinstance(v, tuple):
                    cmd_list += ("--{}".format(k.replace('_', '-')),) + v
                else:
                    cmd_list += (
                        "--{}".format(k.replace('_', '-')),
                        "{}".format(v)
                    )

        return cmd_list

    def run(self, dryrun=False):
        cmd_list = self.build_command()
        print(' '.join(cmd_list))
        if dryrun:
            return "Dryrun"
        else:
            return check_output(cmd_list)
