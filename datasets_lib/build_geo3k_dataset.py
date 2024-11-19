from functools import partial
from torch.utils.data import DataLoader
from .geometry3k_data_loader import Geometry3KDataset, img_train_collator_fn, img_test_collator_fn, img_test_code_gen_collator_fn

def get_geo3k_dataset(args, processor):
    
    if args.experiment == 'code_gen_ft':
        train_dataset = Geometry3KDataset(args, 'train')
        valid_dataset = Geometry3KDataset(args, 'valid')
        test_dataset = Geometry3KDataset(args, 'test')

        collator_test = partial(img_test_code_gen_collator_fn, args=args, processor=processor, device='cuda')
        
        # Train Loader
        train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator_test)
        
        # Valid Loader
        valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator_test)
        
        # Test Loader
        test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator_test)
        
        return train_loader, valid_loader, test_loader
    

    if args.mode == 'schema_head_train':

        print('\n*****Load Train DataLoader*****')
        train_dataset = Geometry3KDataset(args, 'train')
        valid_dataset = Geometry3KDataset(args, 'valid')
    
        print('Schema Header Train')
        
        # Train Loader
        train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size)
        
        # Valid Loader
        valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=args.batch_size)
    
        return train_loader, valid_loader

    elif args.mode == 'schema_head_test':
            
        print('\n*****Load Test DataLoader*****')
        train_dataset = Geometry3KDataset(args, 'train')
        valid_dataset = Geometry3KDataset(args, 'valid')
        test_dataset = Geometry3KDataset(args, 'test')
        
        # Train Loader
        train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size)
        
        # Valid Loader
        valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=args.batch_size)
        
        # Test Loader
        test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size)
        
        return train_loader, valid_loader, test_loader
        
    if args.mode == 'train':
        
        print('\n*****Load Train DataLoader*****')
        train_dataset = Geometry3KDataset(args, 'train')
        valid_dataset = Geometry3KDataset(args, 'valid')
        
        collator = partial(img_train_collator_fn, args=args, processor=processor, device='cuda')
     
        # Train Loader
        train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collator)
        
        # Valid Loader
        valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator)
        
        return train_loader, valid_loader
    
    elif args.mode in ['test', 'viz_attmap']:
        
        print('\n*****Load Test DataLoader*****')
        test_dataset = Geometry3KDataset(args, 'test')
        
        collator = partial(img_test_collator_fn, args=args, processor=processor, device='cuda')
     
        # Test Loader
        test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator)
        
        return test_loader