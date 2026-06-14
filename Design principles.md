# Design principles

Goal:

We are creating a framework that allows end users to emulate their own marketplace. 

The primary use case is to test existing ways of experiment designs in marketplaces, or developing novel ones.



1. We don't code more than needed. We abstract just enough to achieve the goal, but we don't go into a architecture frenzy.
2. Any simulation that happen goes through natural procedure: except the initial setup, nothing happens simulatanously; everything thing is a time driven process; only then we can emulate stock, and dependencies.
3. The framework is configurable, from new entities to new properties, to new actions they can take - all of it is registerable
4. The entities are generic so that seller could be a partner in booking.com's lingo, or a seller in Marktplaats' understanding. A buyer; the same. Etc.
5. Properties of entities are either fixed, or dynamic. Either way, we code in design patterns that treats 'fixed' as a special case of dynamic: a fixed value is set, but there is effective room to define it dynamically. For example, quality may be set once, or via a statistical model
6. The marketplace structure is recursive, so that a market can be part of a bigger market. This allows one to create a market that is a mixture of sub-markets