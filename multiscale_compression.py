import argparse

parser = argparse.ArgumentParser(description='Multiscale Compression')
parser.add_argument('--levels', type=int, default=4)

args = parser.parse_args()
cuda = torch.cuda.is_available()

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if cuda else 'cpu')

flow_dir = join('logs', args.logdir)
assert exists(flow_dir), 'Directory does not exist'.format(flow_dir)
reload_file = join(flow_dir, 'best.tar')
assert exists(reload_file, 'File does not exist'.format(reload_file))

states = torch.load(reload_file)
model.load_state_dict(state['state_dict'])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset = RolloutObservationDataset('datasets/carracing', transform_test,
                                    train=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10,
                                          shuffle=True)
flow_dir = join('logs', args.logdir)
assert exists(flow_dir), 'Directory does not exist'.format(flow_dir)
reload_file = join(flow_dir, 'best.tar')
assert exists(reload_file, 'File does not exist'.format(reload_file))

states = torch.load(reload_file)
model.load_state_dict(state['state_dict'])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset = RolloutObservationDataset('datasets/carracing', transform_test,
                                    train=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10,
                                          shuffle=True)

def process(data):
    data *= 255
    data = torch.floor(data / 64)
    data += torch.rand_like(data)
    return data

data = next(iter(data_loader))
data = process(data)
z, _ = model(data)

total = z.size(1)
zs = []
for level in range(args.levels):
    z_copy = z.copy()
    keep = 0.5 ** level
    if keep < 1
        z_copy[:, :-int(total * keep)] = torch.randn(z.size(0), total * (1 - keep))
    zs.append(z_copy)
z = torch.cat(zs, dim=0)
x = model.sample(z).cpu() / 4

save_image(x, join(flow_dir, 'compression.png'),
           nrow=10)
