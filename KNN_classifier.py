import torch

class KNNClassifier():
    def __init__(self,dims=1, k = 1, label_tags = None):
        '''
        Initialize KNN classifier. This classifier will have
        dimensionality of dims and will look at the k nearest
        neighbors to determine which class to predict.
        Euclidean distance is used to calculate similarity.
        label tags can be used to predict and insert instead
        of numbers if that is more convenient for the user.
        todo: add device option for cuda compatibility.
        :param dims: int, dimensionality of data.
        :param k: int, number of nearest neighbors to use.
        :param label_tags: list of str
        '''
        self.data = torch.empty(0,dims)
        print("Expecting data to have dimensionality: " + str(dims))
        self.labels = []
        self.k = k
        print("Using k = " + str(k))
        
        if label_tags is not None:
            self.set_tags(label_tags)
        else:
            self.label_tags = None
            self.reverse_tags = None
        
    def insert(self,data,label):
        '''
        Insert labeled data point into the classifier.
        Can use either label number or label tag if the
        label_tags have been defined.
        :param data: list, np.array, or torch.Tensor
        :param label: int or str
        '''
        try:
            if 'Tensor' not in str(type(data)):
                data = torch.Tensor([data])
            if len(data.shape) == 1:
                data = torch.reshape(data,[1,data.shape[0]])
        except:
            assert(False),"Invalid data type (Expected torch.Tensor)"

        try:
            assert(isinstance(label,int))
        except AssertionError as e:
            if self.label_tags is None:
                assert(False),"label must be of type int"
            else:
                assert(isinstance(label,str)) ,"label must be of type int or str"
                try:
                    label = self.label_tags[label]
                except KeyError as error:
                    raise(error)
            
        self.data = torch.cat([self.data,data],dim=0)
        self.labels.append(label)
        print('Inserted')
        
        
    def predict(self,data):
        '''
        Use classifier to predict data label.
        Will prioritize using label tags if defined.
        :param data: list, np.array, or torch.Tensor
        '''
        try:
            if 'Tensor' not in str(type(data)):
                data = torch.Tensor([data])
            if len(data.shape) == 1:
                data = torch.reshape(data,[1,data.shape[0]])
        except:
            assert(False),"Invalid data type (Expected torch.Tensor)"
            
        #main algorithm    
        distances = self.data - data
        distances = torch.sum(distances**2,dim=1)
        inds = torch.argsort(distances) #can change this, do we want O(k*n) or O(n log(n))?
        lab = []
        for i in range(min(self.k,inds.shape[0])):
            lab.append(self.labels[inds[i]])
        lab = torch.Tensor(lab).squeeze()
        out = torch.mode(lab)[0]

        if self.reverse_tags is not None:
            return self.reverse_tags[int(out)]
        else:
            return int(out)
            
            
    def set_tags(self,new_tags):
        '''
        Set model tags
        :param new_tags: list of str
        '''
        assert(isinstance(new_tags,list))
        assert(len(new_tags) > 0)
        self.label_tags = {new_tags[i]:i for i in range(len(new_tags))}
        self.reverse_tags = {val:key for key,val in self.label_tags.items()}
        print("Warning: Make sure that incides are correct")
        print(self.label_tags)
        